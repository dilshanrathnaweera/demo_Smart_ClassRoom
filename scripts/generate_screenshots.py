from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
out = Path('data/screenshots')
out.mkdir(parents=True, exist_ok=True)
scenes = [
    ('dashboard_live.png','Live Camera Feed & Occupancy Gauge'),
    ('dashboard_history.png','Occupancy History & Event Timeline'),
    ('dashboard_attendance.png','Attendance View & Export'),
    ('dashboard_upload.png','Uploaded Video Processing & Thumbnails')
]
for name, text in scenes:
    img = Image.new('RGB',(1200,700),(24,30,36))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype('arial.ttf', 36)
    except Exception:
        f = ImageFont.load_default()
    d.text((40,40), text, font=f, fill=(230,240,250))
    # draw sample cards
    for i in range(3):
        x = 40 + i*380
        d.rectangle([x,120,x+340,240], outline=(80,90,100), width=2)
        d.text((x+12,130), f'Card {i+1}', font=f, fill=(200,210,220))
    # thumbnail grid
    tx = 40
    ty = 280
    for r in range(2):
        for c in range(5):
            d.rectangle([tx + c*220, ty + r*120, tx + c*220 + 200, ty + r*120 + 100], outline=(80,90,100))
    img.save(out / name)
    print('wrote', out / name)
print('done')
