from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from bs4 import BeautifulSoup
import time

# Selenium allows us to control the web browser using Python code.
SBR_WEBDRIVER = "https://brd-customer-hl_2f654317-zone-ai_scraper:adumdlpoe535@brd.superproxy.io:9515"

def scrape_website(website):
    print("Connecting to Scraping Browser...")
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, "goog", "chrome")
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        driver.get(website)
        print("Waiting captcha to solve...")
        solve_res = driver.execute(
            "executeCdpCommand",
            {
                "cmd": "Captcha.waitForSolve",
                "params": {"detectTimeout": 10000},
            },
        )
        print("Captcha solve status:", solve_res["value"]["status"])
        print("Navigated! Scraping page content...")
        html = driver.page_source
        return html
    

def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""
    
def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract() #getting rid of tags
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content= "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip())
    return cleaned_content
    
def split_dom_content(dom_content, max_length=6000):
    return [dom_content[i:i+max_length] for i in range(0, len(dom_content), max_length)]






# def scrape_website(website):
#     print("Launching Chrome browser...")

#     # Automatically manage ChromeDriver
#     chrome_driver_path = "./chromedriver"
#     options = webdriver.ChromeOptions()
#     driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)
#     # Set up the actual driver for Chrome browser
#     driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

#     try:
#         driver.get(website)  # Use web driver to go to the site
#         print("Page loaded...")
#         html = driver.page_source  # Get HTML of the page
#         time.sleep(10)  # Wait for the page to load completely

#         return html
#     finally:
#         driver.quit()  # Close the browser
