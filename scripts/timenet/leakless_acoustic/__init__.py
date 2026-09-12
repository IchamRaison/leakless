"""TimeNet connector for the acoustic leak dataset, grouped by dependency cluster."""

from .connector import CONNECTOR, LeaklessAcousticConnector

__all__ = ["CONNECTOR", "LeaklessAcousticConnector"]
