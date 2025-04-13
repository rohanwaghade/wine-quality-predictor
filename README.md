# Wine Quality Predictor

This project predicts the quality of wine using machine learning with a Random Forest Classifier.

## How to Use

### 🐳 Build Docker Image
```bash
docker build -t wine-quality-app .
```

### ▶️ Run Container
```bash
docker run wine-quality-app
```

### 🚀 GitHub Actions CI/CD
Any push to `main` will:
- Run the model
- Test build and generate Docker image