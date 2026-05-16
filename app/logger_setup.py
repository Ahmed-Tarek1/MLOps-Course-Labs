"""
Logging configuration.
"""
import logging
import os
from axiom_py import Client
from axiom_py.logging import AxiomHandler


def setup_logging():
    axiom_client = Client(os.getenv("AXIOM_TOKEN"))

    handler = AxiomHandler(axiom_client, os.getenv("AXIOM_DATASET"))
    handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger, axiom_client