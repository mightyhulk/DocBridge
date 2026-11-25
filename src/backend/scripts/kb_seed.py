"""Re-export seed clinical knowledge from app.kb_seed for standalone scripts."""
from app.kb_seed import CONDITIONS, DRUGS, REFERENCE

__all__ = ["CONDITIONS", "DRUGS", "REFERENCE"]
