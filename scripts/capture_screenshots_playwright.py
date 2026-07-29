"""Capture Streamlit dashboard screenshots (Live, History, Attendance, Upload)

Usage (run locally where Streamlit is reachable):
  pip install playwright
  playwright install chromium
  python scripts/capture_screenshots_playwright.py

This script saves PNGs to `data/screenshots/`.
"""
from playwright.sync_api import sync_playwright
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'screenshots'
OUT.mkdir(parents=True, exist_ok=True)

URL = 'http://localhost:8501'

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1400, 'height': 900})
        print('Loading', URL)
        page.goto(URL, timeout=60000)
        time.sleep(1)

        # Live (home)
        print('Capturing Live')
        page.screenshot(path=str(OUT / 'live.png'), full_page=True)

        # History tab
        try:
            page.click('text=History')
            time.sleep(0.6)
            print('Capturing History')
            page.screenshot(path=str(OUT / 'history.png'), full_page=True)
        except Exception as e:
            print('History tab capture failed:', e)

        # Attendance tab
        try:
            page.click('text=Attendance')
            time.sleep(0.6)
            print('Capturing Attendance')
            page.screenshot(path=str(OUT / 'attendance.png'), full_page=True)
        except Exception as e:
            print('Attendance tab capture failed:', e)

        # Upload UI
        try:
            # return to Live
            page.click('text=Live')
            time.sleep(0.3)
            # select Upload option in sidebar
            page.click('text=Upload Video')
            time.sleep(0.6)
            print('Capturing Upload UI')
            page.screenshot(path=str(OUT / 'upload.png'), full_page=True)
        except Exception as e:
            print('Upload UI capture failed:', e)

        browser.close()
        print('Screenshots written to', OUT)

if __name__ == '__main__':
    main()
