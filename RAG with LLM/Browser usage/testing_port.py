from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")

    print("Connected!")

    print("Contexts:", len(browser.contexts))

    if browser.contexts:
        context = browser.contexts[0]
        print("Pages:", len(context.pages))

    input("Press Enter...")
    
    
# Port is openable if i use : 
'''
& "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" `
--remote-debugging-port=9222 `
--user-data-dir="C:\chrome-debug" `
--profile-directory="Profile 23"
    '''