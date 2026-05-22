from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.routes import compliance_routes, health_routes, query_routes, upload_routes

settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_routes.router, prefix=settings.api_prefix)
app.include_router(query_routes.router, prefix=settings.api_prefix)
app.include_router(upload_routes.router, prefix=settings.api_prefix)
app.include_router(compliance_routes.router, prefix=settings.api_prefix)


@app.get("/")
def home():
    return {"message": "Policy Compliance Assistant Backend Running", "docs": "/docs"}
