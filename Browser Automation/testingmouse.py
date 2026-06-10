import pyautogui, time

print("Hover over any known element in 5 seconds...")
time.sleep(5)
x, y = pyautogui.position()
print(f"X={x}, Y={y}")
# 110 upper # 60 is lower