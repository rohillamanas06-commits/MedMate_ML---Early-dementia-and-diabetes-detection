# 🏥 MedMate - Early Dementia and Diabetes Detection

## About MedMate
MedMate is an advanced machine learning platform designed to provide clinical assessments for two critical conditions: **Diabetes Readmission Risk** and **Early Dementia Detection**. Built with a sleek, modern React frontend and a powerful Flask Python backend, MedMate empowers users and healthcare professionals with instant, data-driven insights.

## 🌟 Key Features

### 🩺 Diabetes Readmission Assessment
- **30-Day Risk Prediction**: Evaluates the likelihood of a patient being readmitted to the hospital within 30 days based on clinical factors.
- **Robust ML Model**: Powered by a robust ensemble model leveraging algorithms like Random Forest, Extra Trees, Gradient Boosting, and SVM.
- **High-Quality Data**: Trained on the extensive **UCI Diabetic Data** (`diabetic_data.csv`), featuring over 100,000 patient records.

### 🧠 Early Dementia Detection
- **Cognitive Assessment**: Detects early signs of dementia based on demographics, clinical history, and cognitive markers.
- **Advanced Ensembling**: Uses a high-accuracy Voting Classifier architecture combined with SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalances.
- **Massive Clinical Dataset**: Trained on the **NACC UDS Dataset** (`investigator_nacc73.csv`), utilizing over 215,000 rows of comprehensive clinical data.

### 💻 Modern User Interface
- **Premium Design**: Features a beautiful dark-mode interface built with Tailwind CSS, Framer Motion, and Shadcn UI.
- **Interactive Forms**: Step-by-step clinical forms with instant feedback and field validations.
- **Visual Result Breakdown**: Presents complex model probabilities in clear, dynamic progress bars and risk level badges.
- **Seamless Navigation**: Fully responsive sidebar navigation and a dedicated demo mode for quick testing.

## 🛠️ Tech Stack

### Frontend
- **React 18** - Component-based UI
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling framework
- **Lucide React** - Beautiful iconography
- **React Router** - Client-side routing

### Backend & Machine Learning
- **Python 3.11+**
- **Flask** - Lightweight web framework
- **Scikit-Learn** - Core machine learning library (Pipelines, VotingClassifiers, SVM, RandomForest)
- **Imbalanced-Learn (SMOTE)** - For handling skewed datasets
- **Pandas & NumPy** - Data processing

## 📁 Project Structure

```bash
 MedMate_ML
 ├── Train_models.py             # Machine learning training pipeline & evaluation
 ├── MedMate_ml.py               # Flask backend API serving the ML models
 ├── requirements.txt            # Python dependencies for backend
 ├── public/                     # Static assets (images, icons)
 ├── src/                        # Frontend Application Source (React)
 │   ├── components/             # Reusable UI components
 │   │   ├── medmate/            # MedMate specific components (Forms, Layout, ResultCard)
 │   │   ├── ui/                 # Shadcn UI primitives
 │   │   └── Footer.jsx          # Application footer
 │   ├── context/                # React Context providers (AuthContext)
 │   ├── lib/                    # API client and configurations
 │   ├── pages/                  # Page components (Landing, Login, Dashboard, etc.)
 │   ├── App.jsx                 # Main application component with routes
 │   ├── main.jsx                # Application entry point
 │   └── index.css               # Global styles and Tailwind directives
 ├── tailwind.config.js          # Tailwind CSS configuration
 ├── vite.config.js              # Vite bundler configuration
 ├── vercel.json                 # Vercel deployment configuration
 ├── package.json                # Node.js dependencies and scripts
 └── README.md                   # Project documentation
```

---
Made with ❤️ by Manas Rohilla
