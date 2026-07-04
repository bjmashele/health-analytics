from fastapi import FastAPI
from api.routers import claim, monitoring, risk
import logging

logging.basicConfig(level=logging.INFO)

logging.info("main.py loaded")
logging.info(f"risk module: {risk}")
logging.info(f"claim module: {claim}")
logging.info(f"monitoring module: {monitoring}")

app = FastAPI(title="Healthcare ML API")


@app.get("/")
def root():
    return {"message": "Healthcare ML API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(risk.router, prefix="/predict", tags=["Risk Score Prediction"])
app.include_router(claim.router, prefix="/predict", tags=["Claim Status Prediction"])
app.include_router(monitoring.router, prefix="/monitor", tags=["Monitoring"])

logging.info("Registered routes:")
for route in app.routes:
    logging.info(route.path)