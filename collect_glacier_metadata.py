import os
import sys
import json
import hashlib
import logging
import requests
from datetime import datetime
from pymongo import MongoClient, UpdateOne

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GlacierMetadataCollector")

# Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "glacier_db"
COLLECTION_NAME = "glacier_records"
OUTPUT_DIR = "output"
BBOX = [74.0, 27.0, 97.0, 37.5] # All India Glaciers Bounding Box

def make_id(source, *parts) -> str:
    raw = f"{source}_{'_'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def fetch_glims_data():
    lon_min, lat_min, lon_max, lat_max = BBOX
    url = "https://www.glims.org/geoserver/ows"
    params = {
        "SERVICE":      "WFS",
        "VERSION":      "1.1.0",
        "REQUEST":      "GetFeature",
        "TYPENAME":     "GLIMS:GLIMS_Glacier_Outlines",
        "BBOX":         f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "MAXFEATURES":  500,
        "OUTPUTFORMAT": "application/json",
    }

    logger.info("Step 1: Fetching glacier features from GLIMS outlines database...")
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        logger.info(f"Successfully retrieved {len(features)} glacier features from GLIMS.")
    except Exception as e:
        logger.error(f"Failed to fetch data from GLIMS WFS: {e}")
        return []

    records = []
    logger.info("Step 2: Processing geometries and mapping property keys...")
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = []
        geom_type = geom.get("type")
        
        # Calculate geographic centroid center for Polygons/MultiPolygons
        if geom_type == "Point":
            coords = geom.get("coordinates", [])
        elif geom_type == "Polygon":
            poly_coords = geom.get("coordinates", [[]])[0]
            if poly_coords:
                lons = [c[0] for c in poly_coords if isinstance(c, (list, tuple)) and len(c) >= 2]
                lats = [c[1] for c in poly_coords if isinstance(c, (list, tuple)) and len(c) >= 2]
                if lons and lats:
                    coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
        elif geom_type == "MultiPolygon":
            try:
                poly_coords = geom.get("coordinates", [[[[]]]])[0][0]
                if poly_coords:
                    lons = [c[0] for c in poly_coords if isinstance(c, (list, tuple)) and len(c) >= 2]
                    lats = [c[1] for c in poly_coords if isinstance(c, (list, tuple)) and len(c) >= 2]
                    if lons and lats:
                        coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
            except Exception:
                pass

        glac_id = props.get("glac_id", "")
        raw_name = props.get("glac_name")
        if not raw_name or raw_name == "None" or raw_name == "Unknown":
            glacier_name = f"Glacier {glac_id}" if glac_id else "Unnamed Glacier"
        else:
            glacier_name = raw_name

        rec = {
            "record_id":     make_id("glims", glac_id, props.get("anlys_id", "")),
            "source":        "GLIMS/RGI",
            "region":        "All India Glaciers",
            "glacier_id":    glac_id,
            "glacier_name":  glacier_name,
            "area_km2":      props.get("db_area") or props.get("area"),
            "longitude":     coords[0] if len(coords) > 0 else None,
            "latitude":      coords[1] if len(coords) > 1 else None,
            "elevation_m":   None,
            "country":       props.get("prim_clas", "India"),
            "analysis_date": props.get("src_date") or props.get("anlys_time", None),
            "fetched_at":    datetime.utcnow().isoformat(),
        }
        records.append(rec)
        
    return records

def fetch_elevations(records):
    # Filter valid coordinates to fetch elevations
    valid_coords = [r for r in records if r["latitude"] is not None and r["longitude"] is not None]
    if not valid_coords:
        return records

    logger.info(f"Step 3: Querying Open-Meteo Elevation API in batches for {len(valid_coords)} records...")
    chunk_size = 100
    for k in range(0, len(valid_coords), chunk_size):
        chunk = valid_coords[k:k+chunk_size]
        lats = ",".join(str(r["latitude"]) for r in chunk)
        lons = ",".join(str(r["longitude"]) for r in chunk)
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            elevations = resp.json().get("elevation", [])
            for r, elev in zip(chunk, elevations):
                r["elevation_m"] = elev
        except Exception as e:
            logger.warning(f"Failed to fetch elevation chunk: {e}")
            
    return records

def save_to_mongodb(records):
    if not records:
        return
    logger.info("Step 4: Upserting collected data records to MongoDB...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]
        col.create_index("record_id", unique=True)
        
        ops = [
            UpdateOne({"record_id": r["record_id"]}, {"$set": r}, upsert=True)
            for r in records
        ]
        result = col.bulk_write(ops)
        logger.info(f"Successfully upserted {result.upserted_count + result.modified_count} records in MongoDB database '{DB_NAME}.{COLLECTION_NAME}'.")
        client.close()
    except Exception as e:
        logger.error(f"Failed to write to MongoDB: {e}")

def save_to_json(records):
    if not records:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "glaciers_metadata.json")
    logger.info(f"Step 5: Saving copy of dataset to {out_path}...")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("JSON file generated successfully.")
    except Exception as e:
        logger.error(f"Failed to write JSON output: {e}")

def main():
    records = fetch_glims_data()
    if not records:
        logger.error("No records collected. Exiting.")
        sys.exit(1)
        
    records = fetch_elevations(records)
    save_to_mongodb(records)
    save_to_json(records)
    logger.info("Glacier Metadata Pipeline Run Finished Successfully.")

if __name__ == "__main__":
    main()
