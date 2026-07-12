<p align="center">
  <img src="__results___files/result_01.png" alt="GlacioWatch — AI Glacier Segmentation Output" width="720"/>
</p>

<h1 align="center">🏔️ GlacioWatch</h1>
<h3 align="center">Automated Feature Detection & Change Analysis for Himalayan Glaciers</h3>
<p align="center">
  <i>Turning raw satellite pixels into life-saving climate intelligence — automatically, every day.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/PyTorch-U--Net-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/MongoDB-GridFS-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/>
  <img src="https://img.shields.io/badge/Firebase-Auth-FFCA28?style=for-the-badge&logo=firebase&logoColor=black" alt="Firebase"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <a href="#-what-is-glaciowatch">Overview</a> •
  <a href="#-the-crisis-behind-the-code">Why It Matters</a> •
  <a href="#-see-it-in-action">Screenshots</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-getting-started">Quick Start</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 🌍 What is GlacioWatch?

Every year, the glaciers that feed the rivers of northern India retreat a little further, and the communities living downstream find out about it a little too late. **GlacioWatch** was built to close that gap.

It's a full, end-to-end platform that watches **29+ major Himalayan glaciers** from space, every single day, without a human ever having to open a satellite image. Underneath the hood it stitches together **remote sensing, deep learning, and modern web engineering** into one autonomous pipeline:

1. A Python pipeline **pulls fresh Sentinel-2 and EnMAP imagery** and cross-references it against curated glacier inventories (GLIMS, WGMS, RGI).
2. A **U-Net deep learning model** segments each image at the pixel level to trace the exact boundary of the ice — no manual tracing required.
3. A **change-detection engine** compares today's boundary against historical baselines to compute retreat rate, and flags glaciers moving into "Rapid Retreat" or "Extreme Melting" territory.
4. A **local LLM (Llama 3.2 via Ollama)** reads the raw numbers and writes a plain-English scientific summary — so the output isn't just a spreadsheet, it's a story a policymaker can actually read.
5. All of it surfaces on a **live, interactive command-center dashboard** built with Next.js, so researchers, disaster-management agencies, and hydropower operators can see glacier health at a glance.

In short: satellites in, insight out — fully automated.

### Key Capabilities

| | |
|---|---|
| 🛰️ **Automated Satellite Ingestion** | Sentinel-2 multispectral + EnMAP hyperspectral imagery, polled in near real time |
| 🧠 **U-Net Semantic Segmentation** | Pixel-level glacier boundary detection at **97.74% accuracy** |
| 🌈 **Hyperspectral Terrain Classification** | 218-band spectral analysis separating Ice, Bare Rock, Water, and Debris |
| 📊 **Automated Change Detection** | Annual retreat-rate calculation with automatic risk classification |
| 🌐 **Live Monitoring Dashboard** | Interactive map covering 29+ Himalayan glaciers, updated from the pipeline in real time |
| 🔔 **Early-Warning Alerts** | Flags glaciers at risk of accelerating retreat or Glacial Lake Outburst Floods (GLOFs) |
| 🤖 **AI-Generated Reports** | A local LLM converts raw metrics into human-readable scientific summaries — zero API cost, zero latency |

---

## 🚨 The Crisis Behind the Code

This project didn't start as a tech demo — it started as a response to a genuinely alarming trend.

| Metric | Impact |
|:---|:---|
| 🧊 **Glacier retreat** | Himalayan glacier-covered area has shrunk by **more than 30%** over the last 40 years |
| 💧 **Water security** | These glaciers feed rivers that over **700 million people** depend on |
| ⚡ **Energy supply** | India's hydroelectric output fell **16.3%** in FY 2023–24, largely due to irregular glacial runoff |
| ⚠️ **Human safety** | The **2023 Sikkim GLOF** — a glacial lake outburst flood — caused a catastrophic flash flood and more than 40 lives lost |

Early warning saves lives and infrastructure. GlacioWatch exists to make that early warning automatic, continuous, and freely visible — instead of something that only happens after a disaster has already made headlines.

---

## 📸 See It In Action

<p align="center">
  <img src="image.png" width="45%"/>
  <img src="image-1.png" width="45%"/>
</p>
<p align="center">
  <img src="image-2.png" width="45%"/>
  <img src="image-3.png" width="45%"/>
</p>

<details>
<summary><b>🔬 Click to see the ML segmentation & spectral analysis outputs</b></summary>
<br/>
<p align="center">
  <img src="__results___files/result_01.png" width="45%"/>
  <img src="__results___files/result_02.png" width="45%"/>
</p>
<p align="center"><i>U-Net glacier segmentation masks overlaid on Sentinel-2 imagery, alongside hyperspectral terrain classification of an alpine test region.</i></p>
</details>

---

## 🏗️ System Architecture

GlacioWatch is deliberately **decoupled** — the ML pipeline, the API, and the web dashboard each do one job well and talk to each other through MongoDB and REST, so any layer can be improved or swapped independently.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            GlacioWatch Architecture                        │
├───────────────┬────────────────┬───────────────────┬───────────────────────┤
│  Data Layer   │   ML Layer     │   Backend / API    │      Frontend         │
├───────────────┼────────────────┼───────────────────┼───────────────────────┤
│ Sentinel-2    │ U-Net          │ FastAPI            │ Next.js 16 (React 19) │
│ EnMAP         │ Segmentation   │ REST endpoints     │ React-Leaflet map     │
│ GLIMS / RGI   │ Random Forest  │ MongoDB + GridFS   │ Framer Motion         │
│ WGMS / SAC    │ Log. Regress.  │ Llama 3.2 (Ollama) │ Tailwind CSS + shadcn │
│ ISRO archives │ Change Detect. │ Firebase Auth + OTP│ Firebase Auth (client)│
└───────────────┴────────────────┴───────────────────┴───────────────────────┘
        │                │                  │                    │
        ▼                ▼                  ▼                    ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                          MongoDB (GridFS)                          │
   │     glaciers_raw · glaciers_processed · pipeline_logs · images    │
   └───────────────────────────────────────────────────────────────────┘
```

**How data flows, end to end:**

`Satellite imagery → Python ML pipeline (U-Net + spectral classifiers) → Change detection → MongoDB → FastAPI / Next.js API → React dashboard → Human decision-maker`

---

## 📈 Technical Achievements

| Metric | Score |
|:---|:---|
| U-Net Pixel Accuracy | **97.74%** |
| U-Net IoU (Intersection over Union) | **0.938** |
| Logistic Regression Balanced Accuracy | **99.68%** |
| Snow/Ice F1 Score | **0.9949** |
| Terrain classes resolved | Ice, Bare Rock, Water, Debris-covered glacier |
| Spectral bands analyzed (EnMAP) | 218 |
| Glaciers actively monitored | 29+ across the Indian Himalaya |
| Pipeline cadence | Fully automated, near real-time Sentinel-2 polling |

---

## 📂 Project Structure

```
Automated-Feature-Detection-and-Change-Analysis/
│
├── src/
│   ├── dataset_pipline/            # Data ingestion & orchestration pipeline
│   │   ├── downloader.py           # Pulls records from GLIMS, WGMS, RGI, curated inventory
│   │   ├── processor.py            # Spectral analysis, enrichment & classification
│   │   ├── change_detector.py      # Area deltas, retreat rate & risk tagging
│   │   ├── db_manager.py           # MongoDB / GridFS persistence layer
│   │   ├── pipeline.py             # Orchestrator — runs the full 9-step pipeline
│   │   ├── scheduler.py            # Runs the pipeline on a recurring schedule
│   │   ├── blog_generator.py       # LLM-powered natural-language report generation
│   │   └── config.py               # Data sources, bounding boxes, region config
│   │
│   ├── backend/                    # FastAPI application server
│   │   ├── main.py                 # API routes — glaciers, alerts, predictions, auth
│   │   ├── firebase_auth.py        # Firebase authentication middleware
│   │   └── email_service.py        # OTP delivery via Gmail SMTP
│   │
│   ├── ml_pipeline/                # Model training & inference
│   │   ├── train.py                # U-Net training script
│   │   ├── predict.py              # Single-image inference
│   │   ├── predict_historical_range.py  # Batch historical predictions
│   │   ├── harvester.py            # Automated satellite tile harvesting
│   │   ├── preprocessor.py         # Image preprocessing & augmentation
│   │   └── ingest_daily_predictions.py  # Ingests daily model outputs into MongoDB
│   │
│   ├── image_data_download/        # Sentinel-2 / Sentinel-1 downloader
│   │   └── data_download.py        # Copernicus Open Access Hub client
│   │
│   ├── ml_inference.py             # Shared inference utilities
│   ├── process_historical_images.py
│   └── automated_pipeline.py       # End-to-end orchestrator (data → ML → DB)
│
├── website/                        # Next.js 16 web dashboard
│   ├── app/
│   │   ├── dashboard/               # Main monitoring command center
│   │   ├── gallery/                 # Research & ML performance gallery
│   │   ├── inventory/                # Searchable glacier inventory
│   │   ├── achievements/             # Model metrics & milestones
│   │   ├── about/ · contact/ · login/
│   │   └── api/                     # Next.js API routes (dashboard, map, alerts, predictions)
│   ├── components/
│   │   ├── sections/                # Landing-page sections (hero, journey, technique…)
│   │   └── ui/                      # shadcn/ui-based design system primitives
│   ├── context/                     # Auth context provider
│   ├── hooks/ · lib/                # Custom hooks & utilities (Firebase, MongoDB, helpers)
│   └── public/images/                # Static research & result imagery
│
├── scalable_glacier_mapping-main/  # Reference research codebase (GlaViTU, DeepLab, SETR, ResUNet)
│   ├── train.py · evaluate.py · deploy.py · predict.py
│   └── dataloaders/ · models/ · layers/ · configs/
│
├── __results___files/              # Notebook-generated visualizations
├── glacier-lack-using-satellite-data.ipynb   # Exploratory data analysis notebook
├── collect_glacier_metadata.py     # Standalone glacier metadata collector
├── process_sentinel_bands.py       # Sentinel band math (NDSI, NDWI, NDVI)
├── run_glacier_predictions.py      # CLI prediction runner
├── test_trained_models.py          # Model testing & validation harness
├── build_report.py                 # Automated PDF report builder
│
├── PROJECT_SUMMARY.md              # Business/strategic project summary
├── LICENSE                         # MIT License
└── README.md                       # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- **MongoDB** (local instance or Atlas cluster)
- **Ollama** *(optional — only needed for local LLM report generation)*
- **Firebase project** *(optional — only needed for authentication)*

### 1. Clone the repository

```bash
git clone https://github.com/jayeshpandey01/Automated-Feature-Detection-and-Change-Analysis.git
cd Automated-Feature-Detection-and-Change-Analysis
```

### 2. Run the data pipeline

```bash
cd src/dataset_pipline
pip install -r requirements.txt

# Copy and fill in environment variables (MongoDB URI, SMTP, Firebase)
cp ../backend/.env.example ../backend/.env

# Run the full 9-step pipeline once
python pipeline.py

# Force a fresh re-download instead of using cached data
python pipeline.py --force

# Or run it forever, on a schedule (every 6 hours)
python scheduler.py --interval 6
```

### 3. Start the FastAPI backend

```bash
cd src/backend
pip install fastapi uvicorn pymongo python-dotenv firebase-admin

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at `http://localhost:8000` — interactive Swagger docs at `http://localhost:8000/docs`.

### 4. Launch the web dashboard

```bash
cd website
npm install

# Create .env.local with your Firebase config, MongoDB URI, and API base URL
# (see website/.env.example for the full list of variables)

npm run dev
```

Open `http://localhost:3000` to see the dashboard.

### 5. (Optional) Pull fresh Sentinel imagery

```bash
cd src/image_data_download
cp .env.example .env
# Fill in your Copernicus Open Access Hub credentials

# One-off metadata pull
python data_download.py --sentinel-only --satellites s1,s2 --days 3

# Continuous polling mode (checks every 15 minutes)
python data_download.py --sentinel-only --satellites s1,s2 --realtime --poll-seconds 900
```

---

## 🧠 The ML Pipeline, In Detail

### U-Net Semantic Segmentation

The core computer-vision model is a **U-Net** trained on Sentinel-2 multispectral tiles to trace glacier boundaries pixel by pixel.

```bash
cd src/ml_pipeline

# Train from scratch
python train.py

# Run inference on a single tile
python predict.py --image path/to/sentinel_tile.tif

# Batch-predict over a historical date range
python predict_historical_range.py --start 2020-01 --end 2026-06
```

A reference research implementation (GlaViTU, DeepLab, SETR, ResUNet architectures) lives in [`scalable_glacier_mapping-main/`](scalable_glacier_mapping-main/) for anyone who wants to experiment with alternative segmentation backbones.

### Hyperspectral Terrain Classification

For finer-grained terrain analysis, **Logistic Regression** and **Random Forest** classifiers run over all **218 EnMAP spectral bands** to separate:

- ❄️ Snow / Ice
- 🪨 Bare Rock
- 💧 Water bodies
- 🏔️ Debris-covered glacier ice

### Automated Change Detection

Every processing cycle, the change-detection engine compares the newly segmented glacier area against its historical baseline and assigns a status:

| Status | Trigger |
|:---|:---|
| 🟢 **Stable** | Area change within normal seasonal variation |
| 🟡 **Rapid Retreat** | Losing more than **2% of area per year** |
| 🔴 **Extreme Melting** | Losing more than **5% of area per year** |

---

## 🌐 The Web Dashboard

Built with **Next.js 16** and **React 19**, the dashboard is designed to feel like a mission-control center rather than a static report:

- 🗺️ **Interactive Glacier Map** — React-Leaflet with CartoDB dark tiles, plotting all monitored glaciers with live status glow-markers
- 📊 **Real-time Statistics** — total glaciers tracked, average area change, and a live risk-level breakdown
- 🖼️ **Research Gallery** — visualized ML performance (accuracy, F1, IoU) pulled straight from pipeline output
- 📋 **Glacier Inventory** — a searchable, sortable, filterable table by state, risk level, and retreat rate
- ✨ **AI-Generated Analysis** — a one-click "Generate Analysis" button that asks a locally-running Llama 3.2 model to turn raw metrics into a readable summary
- 🔐 **Firebase Authentication** — email/password login plus OTP verification via Gmail SMTP for gated features like alert subscriptions

---

## 📡 API Reference

### FastAPI backend (`src/backend/main.py`)

| Endpoint | Method | Description |
|:---|:---|:---|
| `/` | GET | Health check |
| `/glaciers` | GET | List all glaciers, optionally filtered by `region` |
| `/glaciers/{glacier_id}` | GET | Full detail record for one glacier |
| `/alerts` | GET | Currently active high-risk / rapid-retreat alerts |
| `/files/{filename}` | GET | Retrieve a stored pipeline output file |
| `/api/predict/daily` | POST | Run/retrieve the daily model prediction for a glacier |
| `/api/predict/spectral` | POST | Run hyperspectral terrain classification |
| `/api/ingest/raw` | POST | Ingest a new raw data payload from a given source |
| `/api/auth/send-otp` | POST | Send a one-time password to a user's email |
| `/api/auth/verify-otp` | POST | Verify a submitted OTP |
| `/api/auth/firebase-login` | POST | Exchange a Firebase token for a session |
| `/api/auth/notifications/subscribe` | POST | Subscribe an email to alert notifications |
| `/api/auth/trigger-alerts` | POST | Manually trigger the alert-evaluation job |

### Next.js API routes (`website/app/api/`)

| Route | Purpose |
|:---|:---|
| `/api/dashboard` | Aggregated dashboard payload (stats + latest records) |
| `/api/map` | GeoJSON map points for the live glacier map |
| `/api/alerts` | High-risk glacier alerts, formatted for the frontend |
| `/api/predictions` | Model prediction feed for the inventory/gallery views |
| `/api/summary` | AI-generated natural-language summary |
| `/api/image/[id]` | Streams a satellite/result image directly from MongoDB GridFS |

---

## 🗄️ Data Sources

| Source | Coverage | Notes |
|:---|:---|:---|
| **Curated Inventory** | 30 major Indian Himalayan glaciers | Verified coordinates, works fully offline |
| **GLIMS** | Global glacier outlines | Global Land Ice Measurements from Space API |
| **WGMS** | Mass balance & fluctuation data | World Glacier Monitoring Service |
| **RGI** | Randolph Glacier Inventory | Regions 13/14/15 — Central & South Asia |
| **ISRO / SAC** | Himalayan glacier inventory | Space Applications Centre cryosphere data |
| **Sentinel-2 / Sentinel-1** | Multispectral & SAR imagery | Copernicus Open Access Hub |
| **EnMAP** | 218-band hyperspectral imagery | Fine-grained terrain classification |

**Regions monitored:** Karakoram (Siachen, Biafo, Baltoro, Hispar, Rimo) · Ladakh/Zanskar (Drang-Drung, Parkachik, Pensilungpa) · Himachal Pradesh (Bara Shigri, Chhota Shigri, Samudra Tapu) · Uttarakhand (Gangotri, Milam, Pindari, Satopanth, Chorabari) · Sikkim (Zemu, Kangchenjunga, Lhonak, South Lhonak) · Arunachal Pradesh (Kangto, Gorichen)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---|:---|
| **ML / AI** | PyTorch, U-Net, scikit-learn, segmentation-models-pytorch, Ollama (Llama 3.2) |
| **Data Pipeline** | Python, pandas, PyMongo, `schedule`, matplotlib |
| **Backend** | FastAPI, Uvicorn, MongoDB (GridFS), Firebase Admin SDK |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui, Framer Motion |
| **Mapping** | React-Leaflet, CartoDB dark-matter tiles |
| **Auth** | Firebase Authentication, email OTP via Gmail SMTP |
| **Data Sources** | Copernicus Open Access Hub, GLIMS, WGMS, RGI, ISRO/SAC, Sentinel-2, EnMAP |
| **DevOps** | Git, GitHub, MongoDB Atlas / self-hosted MongoDB |

---

## 🔮 Roadmap

- [ ] **SAR integration** — bring in Sentinel-1 radar data so monitoring keeps working straight through cloud cover
- [ ] **Predictive modeling** — LSTM/Transformer forecasting for 48–72 hour GLOF early warning
- [ ] **Edge deployment** — lightweight on-site inference on solar-powered IoT devices near glacial lakes
- [ ] **Multi-temporal decadal analysis** — automated long-range comparison against the Landsat archive
- [ ] **Public research API** — an open, rate-limited API for climate researchers and reinsurers
- [ ] **Mobile companion app** — push notifications and on-the-go risk assessments for nearby communities

---

## 💡 Why This Project Exists

This isn't a toy dataset exercise — it was built to solve real engineering problems: streaming large binary satellite imagery out of MongoDB GridFS through a Next.js API without blowing up memory, keeping an autonomous Python pipeline and a live React frontend in sync, and using a *local* LLM to add genuine interpretive value on top of raw numbers instead of just another dashboard full of charts. If you care about climate tech, applied ML, or full-stack systems that actually ship, there's something here for you.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. If you'd like to help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes with clear messages
4. Open a pull request describing what you changed and why

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ for Climate Science</b><br/>
  <sub>Monitoring the glaciers that sustain 700 million lives — one satellite pass at a time.</sub>
</p>
