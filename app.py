import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import os
import shutil
import subprocess
from skimage.restoration import denoise_tv_chambolle
# import zipfile

import time


# MARK: Page Config
st.set_page_config(
    page_title="JPEGArtifactRemover",
    page_icon="🔍",
    layout="wide"
)

st.markdown(
    """
    <style>

    # [data-testid="stSidebarHeader"] {
    #     display: none;
    # }

    # [data-testid="stSidebarUserContent"] {
    #     padding-top: 1.5rem;
    # }

    [data-testid="stMainBlockContainer"] {
        padding-top: 5rem;
    }
    
    [data-testid="stAppDeployButton"] {
        display: none;
    }

    # [data-testid="stDecoration"] {
    #     display: none;
    # }

    </style>
    """,
    unsafe_allow_html=True
)


# MARK: unsharp_mask()
@st.cache_data(show_spinner=False)
def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Subtract the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# MARK: laplacian_filter()
@st.cache_data(show_spinner=False)
def laplacian_filter(image, sigma=1.0, strength=1.5, kernel_size=(5, 5)):
    # Apply Gaussian blur with specified kernel size
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # Subtract the blurred image from the original
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# MARK: high_pass_filter()
@st.cache_data(show_spinner=False)
def high_pass_filter(image, sigma=1.0):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Subtract the blurred image from the original
    high_pass = cv2.subtract(image, blurred)
    # Add the high-pass image back to the original
    sharpened = cv2.addWeighted(image, 1.0, high_pass, 1.0, 0)
    return sharpened


# MARK: apply_median_blur()
@st.cache_data(show_spinner=False)
def apply_median_blur(image, ksize):
    return cv2.medianBlur(image, ksize)


# MARK: apply_bilateral_filter()
@st.cache_data(show_spinner=False)
def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# MARK: apply_non_local_means()
@st.cache_data(show_spinner=False)
def apply_non_local_means(image, h, template_window_size, search_window_size):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)


# MARK: apply_smoothing_on_gradients()
@st.cache_data(show_spinner=False)
def apply_smoothing_on_gradients(img_rgb,
                                 sobel_ksize=3,             # 3
                                 gradient_threshold=15,     # 30
                                 blur_kernel_size=7):       # 7
    
    # rbat_col1, rbat_col2, rbat_col3, rbat_col4, rbat_col5 = st.columns(5)
    rbat_col2, rbat_col4, rbat_col5 = st.columns(3)

    with st.expander("Show apply_smoothing_on_gradients() function steps"):
        # Перетворюємо зображення в відтінки сірого для аналізу текстури та кольору
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Використовуємо фільтрацію по градієнтах для виявлення зон з плавними переходами
        # Використовує оператор Sobel для обчислення градієнта зображення по осі X. 
        # Це означає, що ми будемо вимірювати зміни інтенсивності пікселів по горизонталі.
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)  # Градієнт по осі X
        # Це аналогічний оператор Sobel, але для осі Y, що вимірює вертикальні зміни інтенсивності.
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)  # Градієнт по осі Y
        # Обчислює модуль градієнтів, який є поєднанням горизонтальних і вертикальних градієнтів. 
        # Він показує, наскільки різко змінюється інтенсивність пікселів у кожному місці зображення. 
        # Величина градієнта використовується для виявлення контурів і переходів кольору.
        grad_mag = cv2.magnitude(grad_x, grad_y)  # Модуль градієнтів
        
        # Маска для визначення областей з малим градієнтом (де немає різких змін кольору)
        # Перетворює результат градієнтів в абсолютні значення і конвертує їх в 8-бітовий формат (від 0 до 255)
        grad_mag = cv2.convertScaleAbs(grad_mag)
        
        st.divider()
        # rbat_col1.image(grad_mag, caption="Gradient Magnitude", use_container_width=True)
        st.image(grad_mag, caption="Gradient Magnitude")

        # Створює бінарну маску, де значення пікселів, що мають градієнт вище порогу 30, отримують значення 0 (чорний), 
        # а пікселі з малим градієнтом (які мають плавні переходи) отримують значення 255 (білий)
        _, mask = cv2.threshold(grad_mag, gradient_threshold, 255, cv2.THRESH_BINARY_INV)  # Чим менший поріг, тим більше зон з плавними переходами

        st.divider()
        rbat_col2.image(mask, caption="Mask for Smooth Areas", use_container_width=True)
        st.image(mask, caption="Mask for Smooth Areas (Білим те що розмивається)")
        
        # Визначення відсотка розмитих та нерозмитих ділянок
        total_pixels = mask.size  # Загальна кількість пікселів (по всьому зображенню)
        smooth_pixels = cv2.countNonZero(mask)  # Кількість білих пікселів (розмитих)
        non_smooth_pixels = total_pixels - smooth_pixels  # Кількість чорних пікселів (нерозмитих)

        # Обчислюємо відсоток
        smooth_percentage = (smooth_pixels / total_pixels) * 100
        non_smooth_percentage = (non_smooth_pixels / total_pixels) * 100

        blur_progress = smooth_percentage / 100

        st.progress(blur_progress)
        st.write(f"Percentage of smoothed area: {smooth_percentage:.2f}%")
        st.write(f"Percentage of non-smoothed area: {non_smooth_percentage:.2f}%")

        tuple_bks = (blur_kernel_size, blur_kernel_size)
        # Створюємо розмиту версію зображення для зон з малим градієнтом
        smoothed_img = cv2.GaussianBlur(img_rgb, tuple_bks, 0)

        st.divider()
        # rbat_col3.image(smoothed_img, caption="Smoothed Image", use_container_width=True)
        st.image(smoothed_img, caption="Smoothed Image")

        # Застосовує маску до розмитого зображення. Це означає, що згладжування буде застосовано лише до тих ділянок, де маска біла (плавні переходи)
        smoothed_img_masked = cv2.bitwise_and(smoothed_img, smoothed_img, mask=mask)

        st.divider()
        rbat_col4.image(smoothed_img_masked, caption="Smoothed Image with Mask", use_container_width=True)
        st.image(smoothed_img_masked, caption="Smoothed Image with Mask (Частина зображення яка розмивається)")

        # Залишаємо незмінними ділянки з великими градієнтами
        non_smoothed_img = cv2.bitwise_and(img_rgb, img_rgb, mask=cv2.bitwise_not(mask))

        st.divider()
        rbat_col5.image(non_smoothed_img, caption="Non-Smoothed Image", use_container_width=True)
        st.image(non_smoothed_img, caption="Non-Smoothed Image (Частина зображення яка не змінюється)")

        # Об'єднуємо результат: плавні зони з розмиттям, інші з без змін
        final_img = cv2.add(smoothed_img_masked, non_smoothed_img)

    return final_img 


# MARK: apply_denoise_tv_chambolle()
@st.cache_data(show_spinner=False)
def apply_denoise_tv_chambolle(image, weight=0.01):

    # Розділення зображення на канали
    channels = cv2.split(image)

    # Виконання Total Variation Denoising для кожного каналу
    denoised_channels = [denoise_tv_chambolle(channel, weight=weight) for channel in channels]

    # Перетворення каналів до формату uint8
    denoised_channels = [(channel * 255).astype(np.uint8) for channel in denoised_channels]

    # Об'єднання каналів назад в кольорове зображення
    tv_denoised_image = cv2.merge(denoised_channels)
    
    return tv_denoised_image


# MARK: convert_jpg_to_png()
# @st.cache_data(show_spinner=False)
def convert_jpg_to_png(uploaded_image, input_folder="uploaded"):

    # Створення директорії uploaded, якщо її немає
    os.makedirs(input_folder, exist_ok=True)
    
    # Збереження завантаженого файлу
    input_path = os.path.join(input_folder, uploaded_image.name)
    output_path = input_path.rsplit(".", 1)[0] + ".png"

    with open(input_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Перевірка, чи вже існує файл у output_path
    if not os.path.exists(output_path):
        # jpeg2png_1.02_x64.exe jpeg2png.exe
        result = subprocess.run(["utils/jpeg2png_1.02_x64.exe", input_path], capture_output=True, text=True) 

        if result.returncode != 0:
            raise RuntimeError("Error converting JPEG to PNG: " + result.stderr)
    
    # Завантаження PNG-зображення назад у програму
    processed_image = Image.open(output_path).convert("RGB")
    return np.array(processed_image)


# MARK: Filter Functions
filter_functions = {
    "Median Blur": apply_median_blur,
    "Bilateral Filter": apply_bilateral_filter,
    "Non-Local Means Denoising": apply_non_local_means,
    "Smoothing on Gradients": apply_smoothing_on_gradients,
    "Denoise TV Chambolle": apply_denoise_tv_chambolle,
}


if "detail_info" not in st.session_state:
    st.session_state.detail_info = True

if "image_comparison_toggle" not in st.session_state:
    st.session_state.image_comparison_toggle = False


# MARK: Sidebar
uploaded_images = st.sidebar.file_uploader("Upload an image", 
                                          type=["jpg", "jpeg"],
                                          label_visibility="collapsed",
                                          accept_multiple_files=True)

# MARK: Crop Image
with st.sidebar.expander("Crop Image", expanded=False, icon="🔍"):
    crop_scale = st.slider("Select crop size (% of the original)", min_value=10, max_value=100, value=100, step=5)

    # Додати вибір позиції кропу
    options = ["Center", "↖️", "↗️", "↙️", "↘️"]
    option_map = {
        "Center": ":material/zoom_in_map:",
        "Top-Left": ":material/north_west:",
        "Top-Right": ":material/north_east:",
        "Bottom-Left": ":material/south_west:",
        "Bottom-Right": ":material/south_east:",
    }
    crop_position = st.segmented_control(
        "Crop position", 
        options=option_map.keys(), 
        format_func=lambda option: option_map[option], 
        selection_mode="single", 
        default="Center",
    )

# MARK: Filtering
st.sidebar.subheader("Filtering", divider="gray")

jpg_to_png_toggle = st.sidebar.toggle(
    label="JPEG Smooth Decoding", 
    value=False, 
    help="Fill in the missing information to create the smoothest transitions in the image")


filtering_methods = st.sidebar.multiselect(
    label="Choose a filtering methods",
    options=list(filter_functions.keys()),
)


with st.sidebar.expander(f"{list(filter_functions.keys())[0]} Settings", 
                        icon="Ⓜ️", 
                        expanded=list(filter_functions.keys())[0] in filtering_methods):

    ksize = st.slider("Select the blur level", 
                    min_value=1, 
                    max_value=9, 
                    value=3, 
                    step=2, 
                    disabled=list(filter_functions.keys())[0] not in filtering_methods,
                    help="Цей параметр визначає розмір області, яка буде використовуватись для обчислення медіанного значення для кожного пікселя."
                        " Кожен піксель в результаті буде замінений на медіанне значення з його сусідніх пікселів (в межах розміру ядра).")

    if st.session_state.detail_info:
        with st.popover(label="Median Blur info", icon="ℹ️", use_container_width=True):
            st.text("Медіанний фільтр — це нелинійний фільтр, який використовується для зменшення шуму на зображенні.\n\n" 
            "Для кожного пікселя обчислюється медіана сусідніх пікселів у вказаному квадратному вікні, і цим значенням замінюється піксель.\n\n" 
            "Це ефективно видаляє шум, зберігаючи краї.")



with st.sidebar.expander(f"{list(filter_functions.keys())[1]} Settings", 
                        icon="🅱️", 
                        expanded=list(filter_functions.keys())[1] in filtering_methods):

    d = st.slider("Diameter of pixel neighborhood", 
    min_value=1, 
    max_value=15, 
    value=9, 
    disabled=list(filter_functions.keys())[1] not in filtering_methods,
    help="Розмір ядра. Розмір області, яка буде використовуватись для обчислення ваги пікселя."
    " Цей параметр визначає діаметр області, яка буде використовуватись для обчислення ваги пікселя.")

    sigma_color = st.slider("Sigma Color", 
    min_value=1, 
    max_value=500, 
    value=75, 
    disabled=list(filter_functions.keys())[1] not in filtering_methods,
    help="Cтандартне відхилення, яке контролює вплив пікселів з різними значеннями інтенсивності кольору."
    "Цей параметр визначає вагу, яка буде використовуватись для обчислення ваги пікселя в просторі кольорів.")

    sigma_space = st.slider("Sigma Space", 
    min_value=1, 
    max_value=500, 
    value=75, 
    disabled=list(filter_functions.keys())[1] not in filtering_methods,
    help="Cтандартне відхилення, яке контролює вплив далеких пікселів на обчислення ваги пікселя."
    "Цей параметр визначає вагу, яка буде використовуватись для обчислення ваги пікселя в просторі пікселів.")

    if st.session_state.detail_info:
        with st.popover(label="Bilateral Filter info", icon="ℹ️", use_container_width=True):
            st.text("Білатеральний (двосторонній) фільтр — це нелінійний фільтр, який одночасно згладжує зображення і зберігає чіткість країв.\n\n"
            "Він враховує не лише відстань між пікселями, а й різницю їх кольорів, щоб уникати розмиття країв.\n\n"
            "Такий фільтр ідеально підходить для зменшення шуму, не втрачаючи важливих деталей.")


with st.sidebar.expander(f"{list(filter_functions.keys())[2]} Settings", 
                        icon=None, 
                        expanded=list(filter_functions.keys())[2] in filtering_methods):

    h = st.number_input(
        label="Filter strength (h)", 
        value=10, 
        step=1,
        help="Цей параметр визначає силу фільтрації. Чим вище значення, тим більше шуму буде видалено, але може виникнути ризик втрати деталей.")

    template_window_size = st.number_input(
        label="Template Window Size", 
        value=7, 
        step=1,
        help="Цей параметр визначає розмір області, яка буде використовуватись для обчислення схожості між пікселями.")
    
    search_window_size = st.number_input(
        label="Search Window Size", 
        value=21, 
        step=1,
        help="Цей параметр визначає розмір області, яка буде використовуватись для пошуку схожих пікселів.")

    if st.session_state.detail_info:
        with st.popover(label="NLM Denoising info", icon="ℹ️", use_container_width=True):
            st.text("Функція fastNlMeansDenoisingColored — це метод усунення шуму, оптимізований для кольорових зображень.\n\n"
            "Вона аналізує схожість пікселів у заданому вікні навколо кожного пікселя та усуває шум, зберігаючи текстуру і деталі.\n\n"
            "Метод ефективно працює на кольорових зображеннях, зменшуючи шум без втрати якості.")


with st.sidebar.expander(f"{list(filter_functions.keys())[3]} Settings", 
                        icon=None, 
                        expanded=list(filter_functions.keys())[3] in filtering_methods):

    sobel_ksize = st.slider(
        label="Sobel Kernel Size", 
        min_value=1,
        max_value=31,
        value=3, 
        step=2,
        disabled=list(filter_functions.keys())[3] not in filtering_methods,
        help="Цей параметр визначає розмір ядра, яке використовується для обчислення градієнтів по осі X та Y.")
    
    gradient_threshold = st.slider(
        label="Gradient Threshold", 
        min_value=0,
        max_value=254,
        value=30, 
        step=1,
        disabled=list(filter_functions.keys())[3] not in filtering_methods,
        help="Цей параметр визначає поріг, за яким визначається яку частину зображення розмивати.")
    
    blur_kernel_size = st.slider(
        label="Blur Kernel Size",
        min_value=1,
        max_value=99, 
        value=7, 
        step=2,
        disabled=list(filter_functions.keys())[3] not in filtering_methods,
        help="Цей параметр визначає розмір ядра, яке використовується для розмиття зображення для зон з малим градієнтом.")

    if st.session_state.detail_info:
        with st.popover(label="info", icon="ℹ️", use_container_width=True):
            st.text("1")


with st.sidebar.expander(f"{list(filter_functions.keys())[4]} Settings", 
                        icon=None, 
                        expanded=list(filter_functions.keys())[4] in filtering_methods):

    weight = st.slider(
        label="Weight", 
        min_value=0.01, 
        max_value=0.15, 
        value=0.03, 
        step=0.01,
        disabled=list(filter_functions.keys())[4] not in filtering_methods,
        help="Цей параметр визначає вагу, яка використовується для обчислення ваги пікселя в просторі кольорів.")

    # esp = st.number_input(
    #     label="esp", 
    #     value=0.0002, 
    #     step=0.0001,
    #     help="поріг для зупинки алгоритму. Коли зміни між двома ітераціями стають меншими за це значення (тобто, зміни стають дуже малими і алгоритм більше не покращує результат), то алгоритм припиняється.")
    
    # max_num_iter = st.number_input( 
    #     label="max_num_iter", 
    #     value=200, 
    #     step=10,
    #     help="Цей параметр вказує максимальну кількість ітерацій, які алгоритм може виконати під час оптимізації.")

    if st.session_state.detail_info:
        with st.popover(label="info", icon="ℹ️", use_container_width=True):
            st.text("1")


# MARK: Sharpening
st.sidebar.subheader("Sharpening", divider="gray")

sharpen_method = []
sharpen_method = st.sidebar.pills(
    label="Sharpening method", 
    options=["Unsharp Masking", "Laplacian Filter", "High-Pass Filter"], 
    selection_mode="single",
    label_visibility="collapsed")

with st.sidebar.expander(
    label="Sharpening Settings", 
    expanded=sharpen_method is not None, 
    icon="🔪"):

    sharpen_sigma = st.slider(
        label="Sigma", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.1, 
        disabled=sharpen_method is None,
        help="Цей параметр визначає стандартне відхилення Гауссівського ядра, яке використовується для розмиття зображення перед відніманням від оригіналу.")

    sharpen_strength = st.slider(
        label="Strength", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.0, 
        step=0.1, 
        disabled=sharpen_method is None or "High-Pass Filter" in sharpen_method,
        help="Цей параметр визначає силу фільтрації. Чим вище значення, тим більше різкість буде підсилена.")

    sharpen_kernel_size = st.slider(
        label="Kernel Size", 
        min_value=1, 
        max_value=15, 
        value=5, 
        step=2, 
        disabled=sharpen_method is None or "Laplacian Filter" not in sharpen_method,
        help="Цей параметр визначає розмір ядра, яке використовується для обчислення другої похідної в кожному пікселі.")

    if st.session_state.detail_info:
        with st.popover(label="Sharpen Methods info", icon="ℹ️", use_container_width=True):
            st.text("Маскування зменшення різкості – це класична техніка, яка передбачає віднімання розмитої версії зображення з вихідного зображення. Це покращує краї та деталі, що призводить до більш чіткого вигляду.\n\n"
            "Фільтр Лапласа — це похідний фільтр другого порядку, який використовується для виявлення країв на зображенні. Застосовуючи фільтр Лапласа, ми можемо виділити краї та посилити загальну різкість.\n\n"
            "Фільтр високих частот є ще одним ефективним методом підвищення різкості зображень. Він працює, дозволяючи високочастотним компонентам (краям і деталям) проходити, одночасно ослаблюючи низькочастотні компоненти (гладкі області)."
            )

# MARK: Обробка
if not uploaded_images:
    st.header("⬅️ Upload an image to get started", divider=False)
else:
    # Почати вимірювання часу
    start_time = time.time()

    # zip_buffer = BytesIO()
    # with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:

    for uploaded_image in uploaded_images:
        
        file_name = uploaded_image.name
        processed_file_name = file_name.replace(".jpg", ".png")

        st.subheader(f"**`{file_name}`**", divider="gray")

        # Читання зображення як PIL Image, конвертація до формату OpenCV
        image = Image.open(uploaded_image)
        
        width, height = image.size
        
        original_size_kb = len(uploaded_image.getbuffer()) / 1024
        
        image_np = np.array(image)

        processed_image = image_np  # Початкове зображення, на яке накладаються фільтри

        if jpg_to_png_toggle:
            processed_image = convert_jpg_to_png(uploaded_image)

        # Перевіряємо, чи вибрано кілька фільтрів і застосовуємо їх по черзі
        for method in filtering_methods:
            if method == list(filter_functions.keys())[0]:
                processed_image = apply_median_blur(processed_image, ksize)
            elif method == list(filter_functions.keys())[1]:
                processed_image = apply_bilateral_filter(processed_image, d, sigma_color, sigma_space)
            elif method == list(filter_functions.keys())[2]:
                processed_image = apply_non_local_means(processed_image, h, template_window_size, search_window_size)
            elif method == list(filter_functions.keys())[3]:
                processed_image = apply_smoothing_on_gradients(processed_image, sobel_ksize, gradient_threshold, blur_kernel_size)
            elif method == list(filter_functions.keys())[4]:
                processed_image = apply_denoise_tv_chambolle(processed_image, weight=weight)

        # Застосування sharpening, якщо вибрано
        if sharpen_method == "Unsharp Masking":
            processed_image = unsharp_mask(processed_image, sigma=sharpen_sigma, strength=sharpen_strength)
        elif sharpen_method == "Laplacian Filter":
            processed_image = laplacian_filter(processed_image, sigma=sharpen_sigma, strength=sharpen_strength, kernel_size=(sharpen_kernel_size, sharpen_kernel_size))
        elif sharpen_method == "High-Pass Filter":
            processed_image = high_pass_filter(processed_image, sigma=sharpen_sigma)

        result_image = Image.fromarray(processed_image)

        # Розрахунок координат для кропу на основі вибраного масштабу та позиції
        width, height = image.size
        crop_width, crop_height = int(width * crop_scale / 100), int(height * crop_scale / 100)
        
        if crop_position == "Center":
            left = (width - crop_width) // 2
            upper = (height - crop_height) // 2
        elif crop_position == "Top-Left":
            left, upper = 0, 0
        elif crop_position == "Top-Right":
            left, upper = width - crop_width, 0
        elif crop_position == "Bottom-Left":
            left, upper = 0, height - crop_height
        elif crop_position == "Bottom-Right":
            left, upper = width - crop_width, height - crop_height
        
        right = left + crop_width
        lower = upper + crop_height
        crop_box = (left, upper, right, lower)

        # Кроп оригіналу та обробленого зображення
        original_crop = image.crop(crop_box)
        processed_crop = result_image.crop(crop_box)

        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_data = buffered.getvalue()

        processed_size_kb = len(img_data) / 1024
        size_difference = processed_size_kb / original_size_kb

        # Додаємо до архіву
        # zip_file.writestr(processed_file_name, img_data)

        if st.session_state.image_comparison_toggle:
            image_comparison(original_crop, processed_crop, "Original", "Processed")

        # MARK: Вивід результатів
        col1, col2 = st.columns(2)

        if not st.session_state.image_comparison_toggle:
            col1.subheader(f"Original Image:")
            col1.image(original_crop, use_container_width=True)

        with col1.popover(label=f"Details (About)", icon="ℹ️", use_container_width=True):
            st.write(f"Resolution: **`{width} x {height} px`**")
            st.write(f"Original image size: **`{round(original_size_kb, 2)} KB`**")
            st.write(f"Processed image size: **`{str(round(processed_size_kb, 2))} KB`** ") 
            st.write(f"Processed image takes **`{round(size_difference, 1)}`** times more space")

        if not st.session_state.image_comparison_toggle:
            col2.subheader("Processed Image:")
            col2.image(processed_crop, use_container_width=True)

        # MARK: Кнопка збереження
        col2.download_button(
            label=f"Save Image **`{processed_file_name}`**",
            data=img_data,
            file_name=processed_file_name,
            mime="image/png",
            icon="⬇️",
            use_container_width=True,
            key=processed_file_name
        )
        
    # Завершення вимірювання часу
    end_time = time.time()
    execution_time = end_time - start_time
    st.sidebar.subheader("Execution Time", divider="gray")
    st.sidebar.write(f"Execution time: **`{execution_time:.2f} seconds`**")

    # zip_buffer.seek(0)  # Переміщуємо вказівник на початок
    # # Кнопка завантаження архіву
    # st.sidebar.download_button(
    #     label="Download All Images as ZIP",
    #     data=zip_buffer,
    #     file_name="processed_images.zip",
    #     mime="application/zip",
    #     icon="⬇️"
    # )


# MARK: Other Settings
st.sidebar.subheader("Other Settings", divider="gray")

if st.sidebar.button(
    label="Clear jpeg smooth decoding cache", 
    icon="🧹" ): 
    shutil.rmtree("uploaded", ignore_errors=True)
    st.sidebar.write("Cache cleared successfully!")

if st.sidebar.button(f"Show Detail Info `{st.session_state.detail_info}`"):
    st.session_state.detail_info = not st.session_state.detail_info
    st.rerun()

if st.sidebar.button(f"Toggle Image Comparison `{st.session_state.image_comparison_toggle}`"):
    st.session_state.image_comparison_toggle = not st.session_state.image_comparison_toggle
    st.rerun()
        
