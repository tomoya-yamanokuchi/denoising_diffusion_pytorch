import os
import numpy as np
from PIL import Image
import imageio

def load_image_with_transparency(path):
    """
    指定したRGB範囲のピクセルを透明にする
    - lower_rgb: (R, G, B) 最小値 (例: #2F2F2F → 47,47,47)
    - upper_rgb: (R, G, B) 最大値 (例: #F3F3F3 → 243,243,243)
    """

    lower_rgb=(45, 45, 45)
    upper_rgb=(48, 48, 48)
    img = Image.open(path).convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        r, g, b, a = item
        if all(lower_rgb[i] <= item[i] <= upper_rgb[i] for i in range(3)):
            # RGBが範囲内なら透明に
            newData.append((r, g, b, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img

def crossfade_images(img1, img2, steps):
    """img1 → img2 にクロスフェードする中間画像を生成"""
    img1_np = np.array(img1).astype(np.float32)
    img2_np = np.array(img2).astype(np.float32)

    frames = []
    for i in range(steps):
        alpha = i / float(steps - 1)
        blended = (1 - alpha) * img1_np + alpha * img2_np
        blended = blended.astype(np.uint8)
        frames.append(Image.fromarray(blended, mode="RGBA"))
    return frames

def generate_transparent_fade_gif(image_folder, output_gif="output.gif",
                                  image_duration=30, transition_frames=10, fps=15):
    """背景透過 + フェード付きGIFを生成"""

    files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(".png")
    ])

    # 背景透過画像を読み込み
    images = [load_image_with_transparency(f) for f in files]

    all_frames = []

    for i in range(len(images)):
        all_frames.extend([images[i]] * image_duration)  # 固定表示

        # フェード作成（最後の画像を除く）
        if i < len(images) - 1:
            fade_frames = crossfade_images(images[i], images[i + 1], transition_frames)
            all_frames.extend(fade_frames)

    # GIF保存
    print(f"Saving GIF with {len(all_frames)} frames to {output_gif}")
    imageio.mimsave(output_gif, all_frames, format="GIF", duration=1 / fps)

# 使い方
path =  "/home/haxhi/workspace/PyBlender_Rendering/hoge/"
generate_transparent_fade_gif(path, f"{path}/fade_transparent.gif")
