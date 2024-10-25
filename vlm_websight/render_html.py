import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def take_screenshot(html_path, image_path):
    # Setup WebDriver options
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--start-maximized")

    try:
        # Initialize WebDriver within a context manager to ensure it closes properly
        with webdriver.Chrome(options=options) as driver:
            driver.get(f"file:///{os.path.abspath(html_path)}")

            total_height = driver.execute_script("""
                return Math.max(
                    document.documentElement.scrollHeight, 
                    document.documentElement.offsetHeight, 
                    document.documentElement.clientHeight,
                    document.body.parentNode.scrollHeight, 
                    document.body.parentNode.offsetHeight, 
                    document.body.parentNode.clientHeight
                );
            """)

            # Set the browser window to capture the whole page
            driver.set_window_size(1280, total_height)  # Width can be fixed, height should be total height of the page
            driver.save_screenshot(image_path)
            driver.quit()

        print(f"Saved screenshot at: {image_path}")
    except Exception as e:
        print(f"Failed to render {html_path}: {str(e)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--html_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    take_screenshot(args.html_path, args.image_path)