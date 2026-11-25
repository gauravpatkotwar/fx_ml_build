"""Health checks for Render"""
from fastapi import APIRouter

router = APIRouter()

@router.get('/health')
async def health_check():
    return {'status': 'healthy', 'service': 'fx-ml-trading'}

@router.get('/ready')
async def readiness_check():
    return {'status': 'ready'}
