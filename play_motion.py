import mujoco
import cv2
import pandas as pd
import numpy as np

# 1. Загружаем файл движения
motion_data = pd.read_csv('motion.csv')

# 2. Путь к 3D модели робота G1 из скачанного репозитория
# (Проверь этот путь в своей папке, он может немного отличаться)
model_path = 'resources/robots/g1/g1.xml' 

try:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}. Проверь путь к файлу g1.xml")
    exit()

# 3. Настройка записи видео
scene = mujoco.MjvScene(model, maxgeom=10000)
cam = mujoco.MjvCamera()
option = mujoco.MjtOption()
pert = mujoco.MjvPerturb()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Разрешение видео
width, height = 640, 480
video_writer = cv2.VideoWriter('motion.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

print("Запуск симуляции и запись видео...")

# 4. Цикл проигрывания кадров из CSV
for index, row in motion_data.iterrows():
    # Устанавливаем позиции суставов из колонок CSV
    # Формат LAFAN обычно передает кватернионы или углы, сопоставляем их с qpos робота
    for i in range(min(model.nq, len(row))):
        data.qpos[i] = row.iloc[i]
        
    mujoco.mj_step(model, data)
    
    # Рендерим кадр
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(model, data, option, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    # Читаем пиксели экрана
    rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb_buffer, None, viewport, context)
    bgr_frame = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2BGR)
    bgr_frame = cv2.flip(bgr_frame, 0) # переворачиваем кадр
    
    video_writer.write(bgr_frame)

video_writer.release()
print("Готово! Видео motion.mp4 успешно сохранено в папку проекта.")
