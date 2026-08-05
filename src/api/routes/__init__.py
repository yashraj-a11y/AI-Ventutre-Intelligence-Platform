"""Day 14 — API routers, collected for inclusion in the app."""
from src.api.routes import (analyze, counterfactual, explain, health, predict,
                            recommend, report, retrieve, scenario)

routers = [health.router, predict.router, explain.router, retrieve.router,
           counterfactual.router, scenario.router, recommend.router, analyze.router,
           report.router]
