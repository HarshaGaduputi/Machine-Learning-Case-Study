"""Launch the case-study dashboard with Streamlit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dashboard import main  # noqa: E402


main()
