# CoreForge

<p align="center">
  <img src="assets/coreforge.png" width="160" height="160" alt="CoreForge">
</p>

<p align="center">
  <b>Стресс-тест GPU для Windows</b> — NVIDIA и AMD
</p>

Нагружает compute-ядра, тензорные ядра (NVIDIA), шейдеры, RT, видеопамять, Copy-движок и видеокодировщик. Блоки можно гонять **вместе** или **по очереди**. Интерфейс на русском.

Сторонние Python-пакеты для запуска **не нужны**: CUDA, Vulkan, OpenGL и ADL вызываются через `ctypes` из драйвера.

## Возможности

| Блок | NVIDIA | AMD | Как устроено |
|---|---|---|---|
| **Compute / CUDA** | да | да (Vulkan compute) | PTX FMA/SFU/INT на NVIDIA, Vulkan compute на AMD |
| **Тензорные ядра** | да | нет | `mma.sync` FP16 / TF32 / FP8 (Ada), несколько независимых цепочек |
| **Шейдеры** | да | да | OpenGL 4.3: читаемые сцены + compute |
| **FSR 1** | да | да | EASU + RCAS: рендер в меньшем разрешении, апскейл, FPS натив / FSR |
| **RT-ядра** | да | да (RDNA 2+) | Vulkan ray query + TLAS |
| **VRAM** | да | да | занятие объёма, копирование, random walk |
| **Copy** | да | да | CUDA DMA или `vkCmdCopyBuffer` |
| **NVENC / NVDEC** | да | нет | отдельный процесс, не роняет окно |
| **Датчики** | NVML + NvAPI | ADL (Adrenalin) | t°, hotspot, VRAM, Вт, частоты, вентилятор |

На **AMD** галочки CUDA, тензоров, NVENC и NVDEC выключены — этого железа нет. Шейдеры, VRAM, Copy и RT (RDNA 2+) остаются.

Режим **вместе** гоняет выбранные блоки параллельно. **По очереди** — по одному, с длительностью шага.

Есть лимит температуры GPU, интенсивность, разрешение окна шейдеров, FSR 1 (пресеты Ultra Quality / Quality / Balanced / Performance), бесконечный прогон (стоп вручную или по t°).

Закрытие окна **без СТОП** гасит все потоки и процесс кодировщика — нагрузка не остаётся в фоне.

## Сцены шейдеров

Окно `CoreForge — шейдеры` каждые ~14 с меняет сцену. Картинка строится raymarch’ем с нормалями и освещением:

1. **туннель** — кольца и направляющие в тёмной трубе
2. **фрактал** — Mandelbox-подобная структура
3. **сияние** — полярное сияние и звёзды
4. **город** — кварталы, окна, закатное небо
5. **океан** — волны и блик
6. **кристаллы** — грани вокруг камеры
7. **плазма** — цветовые поля

Сверху оверлей: частоты GPU/MEM, t° GPU / hotspot / VRAM, Вт, загрузка, занятая память, сцена, FPS. Если включён FSR 1 — разрешение рендера → выход и FPS натив / после апскейла.

## Требования

- Windows 10/11 x64
- Python 3.10+ **или** готовый `CoreForge.exe`
- Актуальный драйвер видеокарты

**NVIDIA:** Game Ready / Studio. Нужны `nvcuda.dll`, `nvml.dll`. Для RT — `vulkan-1.dll`.  
**AMD:** Adrenalin. Нужны `atiadlxx.dll` и Vulkan Runtime (часто ставится с драйвером). RT — на RDNA 2 и новее.

Ссылки:

- [Python](https://www.python.org/downloads/windows/) — отметьте *Add python.exe to PATH*
- [Драйвер NVIDIA](https://www.nvidia.com/Download/index.aspx?lang=ru)
- [Драйвер AMD](https://www.amd.com/en/support/download/drivers.html)
- [Vulkan Runtime](https://vulkan.lunarg.com/sdk/home#windows)

## Запуск

### Готовый exe

1. Соберите `build_exe.bat` (нужен Python один раз).
2. Запускайте **`dist\CoreForge\CoreForge.exe`**.
   Рядом должны лежать файлы из этой папки `dist\CoreForge` — не вытаскивайте один exe.
3. Папка `build` — черновик PyInstaller. Exe оттуда **не запускать** (ошибка `python314.dll`).
4. Иконка вшита в exe: на панели задач и у ярлыка один и тот же знак.

После сборки создаются ярлыки `CoreForge.lnk` в папке проекта и на рабочем столе. Вручную: `create_shortcut.bat`.

### Из исходников

```bat
setup.bat
run.bat
```

или:

```bat
python main.py
```

`run.bat` стартует через `pythonw`, без чёрной консоли. Иконка окна и панели задач берётся из `assets/coreforge.ico`.

## Как пользоваться

1. Выберите **Вместе** или **По очереди**.
2. Включите нужные блоки. На AMD тензор и NVENC/NVDEC серые.
3. Задайте объём VRAM (запас ~0.9 ГБ системе), интенсивность, лимит t°.
4. По желанию включите **FSR 1** и разрешение рендера.
5. **Бесконечно** — по умолчанию, стоп кнопкой или по температуре.
6. **ЗАПУСК**. Смотрите графики, лог и окно шейдеров.

Чтобы увидеть нагрузку в диспетчере задач:

- 3D / шейдеры — галочка **Шейдеры**
- Compute — **CUDA** и **Тензор**
- Copy — **Copy-движок**
- Video Encode / Decode — **NVENC / NVDEC** (NVIDIA)
- RT — **RT-ядра**, счётчик «Млучи/с» в шапке

Полные ватты ближе к TGP, если включены compute + тензор + шейдеры + VRAM + Copy + RT.

## Сборка exe для GitHub Releases

На машине с Python 3.10+:

```bat
build_exe.bat
```

Результат: `dist\CoreForge\CoreForge.exe` (onedir, надёжнее onefile для ctypes/драйверов). Иконка: `assets/coreforge.ico`.

Не запускайте ничего из `build\` — рабочий файл только в `dist\CoreForge\`.

В Release выложите zip папки `dist\CoreForge`. Пользователю Python не нужен.

## Безопасность

Ядра короткие (~70 мс), чтобы не срабатывал Windows TDR. При достижении лимита температуры тест останавливается. Не оставляйте ПК без присмотра на максимальной мощности. Автор не отвечает за повреждения железа.

## Структура

```
main.py                 точка входа
run.bat                 запуск без консоли
setup.bat               проверка Python и драйверов
coreforge/
  engine.py             оркестрация блоков
  ui.py                 интерфейс
  gl_stress.py          окно шейдеров
  fsr1.py               FSR 1 EASU + RCAS
  rt_stress.py          Vulkan RT
  vk_compute.py         compute/VRAM/Copy для AMD
  cuda_api.py           CUDA Driver API
  ptx.py                PTX-ядра
  copy_stress.py        NVIDIA DMA
  media_*.py            NVENC/NVDEC
  nvml_api.py           датчики NVIDIA
  adl_api.py            датчики AMD
  vendor.py             определение вендора
  assets.py             пути к иконке
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
