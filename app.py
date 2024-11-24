import streamlit as st
# from streamlit_image_comparison import image_comparison

import cv2

from PIL import Image
import numpy as np

from io import BytesIO

import os
import shutil
import subprocess




# MARK: Page Config
st.set_page_config(
    page_title="OSZ Coursework",
    page_icon="🔍",
    layout="wide"
)


st.markdown(
    """
    <style>
    [data-testid="stSidebarHeader"] {
        display: none;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3rem;
    }
    [data-testid="stAppDeployButton"] {
        display: none;
    }
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







filter_functions = {
    "Median Blur": apply_median_blur,
    "Bilateral Filter": apply_bilateral_filter,
    "Non-Local Means Denoising": apply_non_local_means
}






if "detail_info" not in st.session_state:
    st.session_state.detail_info = True






# MARK: Sidebar
uploaded_image = st.sidebar.file_uploader("Upload an image", 
                                          type=["jpg", "jpeg"],
                                          label_visibility="collapsed")


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










st.sidebar.subheader("About image", divider="gray")


# MARK: Обробка
if uploaded_image is None:
    st.header("⬅️ Upload an image to get started", divider=False)

    st.sidebar.write("Upload an image first.")
else:

    

    # Читання зображення як PIL Image, конвертація до формату OpenCV
    image = Image.open(uploaded_image)

    width, height = image.size
    st.sidebar.write(f"Resolution: **`{width} x {height} px`**")

    original_size_kb = len(uploaded_image.getbuffer()) / 1024
    st.sidebar.write(f"Original image size: **`{round(original_size_kb, 2)} KB`**")

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


    # MARK: Вивід результатів
    col1, col2 = st.columns(2)

    with col1:
        # with st.container(border=True):
        st.subheader("Original Image:")
        # st.image(image, use_container_width=True)
        st.image(original_crop, use_container_width=True)

    with col2:
        # with st.container(border=True):
        st.subheader("Processed Image:")
        # st.image(sharpened_image, use_container_width=True)
        st.image(processed_crop, use_container_width=True)


        # MARK: Кнопка збереження
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        img_data = buffered.getvalue()

        processed_size_kb = len(img_data) / 1024
        size_difference = processed_size_kb / original_size_kb
        st.sidebar.write(f"Processed image size: **`{str(round(processed_size_kb, 2))} KB`** ") 
        st.sidebar.write(f"Processed image takes **`{round(size_difference, 1)}`** times more space")

        # Додати кнопку для завантаження обробленого зображення
        st.download_button(
            label="Save Image",
            data=img_data,
            file_name="processed_image.png",
            mime="image/png",
            icon="⬇️",
            use_container_width=True
        )





# MARK: Other Settings
st.sidebar.subheader("Other Settings", divider="gray")

if st.sidebar.button(
    label="Clear jpeg smooth decoding cache", 
    icon="🧹"
    ): 

    shutil.rmtree("uploaded", ignore_errors=True)
    st.sidebar.write("Cache cleared successfully!")


if st.sidebar.button(f"Show Detail Info `{st.session_state.detail_info}`"):

    st.session_state.detail_info = not st.session_state.detail_info
    st.rerun()
    
