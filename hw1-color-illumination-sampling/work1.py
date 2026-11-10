import cv2
import numpy as np
import sys
print("Python:", sys.version.split()[0], "| OpenCV:", cv2.__version__, "| NumPy:", np.__version__)

#-----------------------------------------------------------------------------------------------------------------------
# PREDICTION:
# In the shadow photo, the R, G and B values should drop because less light reaches the scene.
# In the normal photo, the yellowish room light should make R and G higher, while in the light photo the white light
# should raise the values and make the blue channel higher.
#-----------------------------------------------------------------------------------------------------------------------

PATHS = {"normal": "pictures/normal.jpeg",
         "golge": "pictures/shadow.jpeg",
         "lamba": "pictures/light.jpeg"}

imgs_bgr = {}
for name, path in PATHS.items():
    img = cv2.imread(path)
    if img is  None:
        raise FileNotFoundError(f"File not found:{path}")

    imgs_bgr[name] = img

normal = imgs_bgr["normal"]

print(f"shape normal: {normal.shape}\n"
      f"dtype: {normal.dtype}\n"
      f"min: {normal.min()}, max:{normal.max()}")

for name, img in imgs_bgr.items():
    print(f"name: {name}\n"
          f"shape: {img.shape}\n")

imgs_rgb = {}
for name, img in imgs_bgr.items():
    imgs_rgb[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Adım 4c: img[y, x, kanal] erişimi; OpenCV BGR okur, RGB'de kanal sırası tersine döner
print("img[400, 690] BGR:", imgs_bgr["normal"][400, 690], "| RGB:", imgs_rgb["normal"][400, 690])

POINTS = {"nesne": (690, 400), "kagit": (550, 1400), "masa": (1000, 900)}
R = 7

def patch_mean(rgb, x, y, r=R):
    patch = rgb[y-r : y+r+1, x-r : x+r+1]
    return patch.reshape(-1, 3).mean(axis=0)

for name, rgb in imgs_rgb.items():
    for point_name, (x, y) in POINTS.items():
        m = patch_mean(rgb, x, y)
        print(name, point_name, (x, y), m.round(0))

# PREDICTION 2: In the shadow photo, the V channel should drop while S should stay about the same; in the light photo,
# V should rise, S should drop, and H may shift slightly because the colour of the light changes.
# I expect the fixed RGB mask to shrink in the shadow and light photos, because all three channels move together
# and leave the frozen range. I expect the HSV mask to be affected less, because H and S depend less on brightness.

imgs_hsv = {}
for name, rgb in imgs_rgb.items():
    imgs_hsv[name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

x, y = 700, 170
for name in imgs_rgb:
    print(name, "RGB", patch_mean(imgs_rgb[name], x, y).round(0),
          "HSV", patch_mean(imgs_hsv[name], x, y).round(0))

rgb_lo, rgb_hi = np.array([80, 83, 17]), np.array([140, 143, 77])
hsv_lo, hsv_hi = np.array([23, 102, 63]), np.array([39, 202, 163])

for name in imgs_rgb:
    m_rgb = cv2.inRange(imgs_rgb[name], rgb_lo, rgb_hi)
    m_hsv = cv2.inRange(imgs_hsv[name], hsv_lo, hsv_hi)
    oran_rgb = np.count_nonzero(m_rgb) / m_rgb.size
    oran_hsv = np.count_nonzero(m_hsv) / m_hsv.size
    print(name, "RGB oran:", round(oran_rgb, 4), "HSV oran:", round(oran_hsv, 4))
    cv2.imwrite(f"mask_rgb_{name}.png", m_rgb)
    cv2.imwrite(f"mask_hsv_{name}.png", m_hsv)

# PREDICTION 3: Adding +40 should raise the mean brightness by about 40, slightly less because some pixels clip at 255.
# Gamma 0.5 should brighten the image (mean goes up) and gamma 2.0 should darken it (mean goes down).

def shift_brightness(img, b):
    out = img.astype(np.float32) + b
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)

def gamma_transform(img, gamma):
    out = 255.0 * (img.astype(np.float32) /255) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)

base = imgs_rgb["normal"]
print("orijinal ortalama:", base.mean())
for label, res in [("b=+40", shift_brightness(base, 40)),
                   ("gamma 0.5", gamma_transform(base, 0.5)),
                   ("gamma 2.0", gamma_transform(base, 2.0))]:
    print(label, res.mean())

kirpilan = np.count_nonzero(base.astype(np.int32) + 40 > 255)
print("255'te kırpılan değer sayısı:", kirpilan)

for label, res in [("shift40", shift_brightness(base, 40)),
                   ("gamma05", gamma_transform(base, 0.5)),
                   ("gamma20", gamma_transform(base, 2.0))]:
    cv2.imwrite(f"result_{label}.png", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

# PREDICTION 4: With INTER_NEAREST, thin lines and handwriting should break and create false patterns after 1/4 downscaling.
# With INTER_AREA, the text should turn grey and blurry without false patterns. Enlarging back should not restore the lost detail.

x, y, s = 330, 700, 400
crop = imgs_rgb["normal"][y:y+s, x:x+s]

kucuk_n = cv2.resize(crop, (s//4, s//4), interpolation=cv2.INTER_NEAREST)
kucuk_a = cv2.resize(crop, (s//4, s//4), interpolation=cv2.INTER_AREA)

buyuk_n = cv2.resize(kucuk_n, (s, s), interpolation=cv2.INTER_NEAREST)
buyuk_a = cv2.resize(kucuk_a, (s, s), interpolation=cv2.INTER_NEAREST)

yan_yana = np.hstack([crop, buyuk_n, buyuk_a])
cv2.imwrite("result/kesit_karsilastirma.png", cv2.cvtColor(yan_yana, cv2.COLOR_RGB2BGR))