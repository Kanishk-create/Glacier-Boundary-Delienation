# 🏔️ GlacioWatch: Himalayan Glacier Monitoring System

## 🌟 Project Vision
GlacioWatch is an autonomous, end-to-end platform designed to monitor the health and retreat of Himalayan glaciers in real-time. By bridging remote sensing, machine learning, and modern web engineering, it translates raw satellite data into actionable climate science insights and early warning signals for Glacial Lake Outburst Floods (GLOFs).

---

## 🏗️ System Architecture

The project is built as a decoupled, data-driven ecosystem consisting of four main layers:

### 1. Data Pipeline (Python & ML)
Located in `src/dataset_pipline`, this layer handles the "heavy lifting" of Earth Observation data:
- **Satellite Ingestion**: Automated downloading of Sentinel-2 and EnMAP hyperspectral imagery focusing on the Himalayan ranges.
- **AI Segmentation (U-Net)**: A high-accuracy U-Net model performs pixel-level glacier segmentation (97.74% Accuracy, 0.938 IoU).
- **Hyperspectral Analysis**: Uses Logistic Regression and Random Forest on 218 spectral bands for fine-grained terrain classification (Ice, Bare Rock, Water, Debris).
- **Change Detection**: Calculates annual retreat rates and identifies "Rapid Retreat" (>2%/yr) or "Extreme Melting" (>5%/yr) status.
- **Persistence**: Results, metrics, and high-res processed images are stored in **MongoDB** (using GridFS for large binaries).

### 2. Backend API (FastAPI & Next.js)
- **FastAPI**: Serves real-time statistics, historical trends, and high-risk alerts.
- **Next.js 15 API Routes**: Streams live imagery directly from MongoDB GridFS and manages dashboard state.
- **Local LLM Integration**: Uses **Llama 3.2:1b** (via Ollama) to synthesize raw metrics into human-readable scientific summaries automatically.

### 3. Web Dashboard (Next.js + Tailwind)
A "Neon-Dark" Command Center experience:
- **Framework**: Next.js 15 with Framer Motion for high-fidelity animations.
- **Interactive Map**: React-Leaflet integration with CartoDB tiles plotting 29 major glaciers with live status indicators.
- **Research Gallery**: Dynamic display of ML performance metrics (AUC, F1, Accuracy) for transparency in climate reporting.

### 4. Mobile Application (React Native / Expo)
Located in `neesh`, providing:
- Real-time monitoring notifications.
- Risk level assessments for nearby communities.
- On-the-go access to glacier health trends.

---

## 📉 The Crisis & Impact
- **Retreat**: Himalayan glacier area has decreased by >30% in 40 years.
- **Water Security**: Threatens water supply for over 700 million people.
- **Energy**: India's hydroelectric output dropped 16.3% in FY 23-24 due to irregular glacial runoff.
- **Safety**: The 2023 Sikkim GLOF highlighted the desperate need for early warning systems.

---

## 💰 Strategic Value & Business Model
GlacioWatch is positioned at the intersection of ClimateTech and Disaster Risk Reduction (DRR):
- **Primary Market**: Hydropower operators (NHPC, SJVN) seeking to protect multi-billion dollar infrastructure.
- **Government Integration**: Aligning with India's NDMA $20M risk mitigation programmes.
- **Revenue Streams**: B2B Subscriptions for hydro-operators, Government licensing for disaster management, and API data licensing for reinsurers.

---

## 🛠️ Technical Achievements
| Metric | Achievement |
| :--- | :--- |
| **U-Net Accuracy** | 97.74% |
| **U-Net IoU** | 0.938 |
| **Logistic Regression BA** | 99.68% |
| **Snow/Ice F1 Score** | 0.9949 |
| **Processing Time** | Fully automated pipeline with real-time polling |

---

## 🚀 Future Roadmap
1. **SAR Integration**: Incorporating Sentinel-1 Radar data to monitor glaciers through cloud cover.
2. **Predictive Modeling**: Using LSTM/Transformers to forecast GLOF events 48-72 hours in advance.
3. **Edge Deployment**: Running lightweight classification models on-site using solar-powered edge devices.
