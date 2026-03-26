import streamlit as st
import google.generativeai as genai
import replicate
from rembg import remove
from PIL import Image
import io

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Product Studio Pro", page_icon="📸", layout="wide")

st.title("📸 AI Product Photo Studio")
st.markdown("Sulap foto produk biasa menjadi aset visual profesional siap iklan.")
st.divider()

# --- 2. SIDEBAR & MANAJEMEN API KEY ---
with st.sidebar:
    st.header("⚙️ Konfigurasi Mesin AI")
    st.markdown("Masukkan API Key Anda untuk mengaktifkan mesin render.")
    
    # Mengambil API key dari input user (atau bisa dari st.secrets nantinya)
    gemini_key = st.text_input("Google Gemini API Key:", type="password")
    replicate_key = st.text_input("Replicate API Key:", type="password")
    
    st.divider()
    theme_choice = st.selectbox(
        "Pilih Tema Latar Belakang:", 
        [
            "Commercial Studio (Terang & Bersih)", 
            "Alam/Outdoor Estetik (Sinar Matahari)", 
            "Dark & Elegant (Premium/Mewah)"
        ]
    )

    theme_prompts = {
        "Commercial Studio (Terang & Bersih)": "placed on a clean white studio podium, bright professional lighting, 8k, photorealistic",
        "Alam/Outdoor Estetik (Sinar Matahari)": "placed on a wooden table outdoors, blurred green nature background, warm sunlight, golden hour, cinematic",
        "Dark & Elegant (Premium/Mewah)": "placed on a black marble surface, dark moody lighting, elegant aesthetics, highly detailed, dramatic shadows"
    }

# --- 3. FUNGSI PEMBANTU (BACKGROUND REMOVAL) ---
@st.cache_data(show_spinner=False)
def remove_background(image_bytes):
    """Menghapus latar belakang gambar menggunakan library rembg"""
    result = remove(image_bytes)
    return Image.open(io.BytesIO(result)).convert("RGBA")

# --- 4. MAIN WORKSPACE ---
col1, col2 = st.columns(2)

# KOLOM KIRI: INPUT & AUTOSCAN
with col1:
    st.subheader("1. Unggah & Ekstrak Produk")
    uploaded_file = st.file_uploader(
        "Pilih foto produk (Contoh: Kemasan cokelat SILVERWIN, botol skincare, dll)", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Tampilkan gambar original
        image = Image.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
        st.image(image, caption="Foto Original", use_container_width=True)
        
        # Tombol Autoscan & Hapus Latar
        if st.button("🔍 Jalankan Autoscan & Hapus Latar"):
            if not gemini_key:
                st.error("⚠️ Masukkan Gemini API Key terlebih dahulu di sidebar!")
            else:
                with st.spinner("AI sedang menganalisis DNA produk dan menghapus latar belakang..."):
                    try:
                        # 1. Hapus Latar Belakang
                        bg_removed_img = remove_background(image_bytes)
                        st.session_state['bg_removed_img'] = bg_removed_img
                        
                        # 2. Analisis Vision menggunakan Gemini
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        system_prompt = """
                        Analisis gambar produk ini. Ekstrak elemen fisik utamanya ke dalam 
                        format kata kunci bahasa Inggris yang dipisahkan dengan koma. 
                        Fokus HANYA pada bentuk produk, warna, tekstur, dan teks merek.
                        Berikan HANYA kata kuncinya tanpa kalimat pengantar.
                        """
                        
                        response = model.generate_content([system_prompt, image])
                        st.session_state['scan_result'] = response.text.strip()
                        
                        st.success("Autoscan & Pemotongan Latar Selesai!")
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")

# KOLOM KANAN: RENDER ENGINE (REPLICATE)
with col2:
    st.subheader("2. Mesin Render Final")
    
    if 'scan_result' in st.session_state and 'bg_removed_img' in st.session_state:
        st.image(st.session_state['bg_removed_img'], caption="Produk Transparan (Siap Render)", width=200)
        
        final_prompt = f"{st.session_state['scan_result']}, {theme_prompts[theme_choice]}"
        st.info(f"**Prompt Rahasia:**\n{final_prompt}")
        
        if st.button("✨ Render Foto Profesional"):
            if not replicate_key:
                st.error("⚠️ Masukkan Replicate API Key terlebih dahulu di sidebar!")
            else:
                with st.spinner(f"Merender tema {theme_choice} menggunakan Stable Diffusion..."):
                    try:
                        # Setup Replicate API
                        import os
                        os.environ["REPLICATE_API_TOKEN"] = replicate_key
                        
                        # Mengubah gambar PIL transparan menjadi file sementara untuk dikirim ke API
                        buf = io.BytesIO()
                        st.session_state['bg_removed_img'].save(buf, format='PNG')
                        buf.seek(0)
                        
                        # Memanggil model Stable Diffusion Image-to-Image (SDXL) di Replicate
                        # Catatan: Endpoint model ini disesuaikan untuk img2img
                        output = replicate.run(
                            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                            input={
                                "image": buf,
                                "prompt": final_prompt,
                                "prompt_strength": 0.8, # Menjaga bentuk asli produk
                                "num_inference_steps": 30,
                                "negative_prompt": "text distortion, bad proportions, ugly, disfigured"
                            }
                        )
                        
                        # Menampilkan hasil akhir
                        if output:
                            final_image_url = output[0]
                            st.image(final_image_url, caption="Hasil Render Final", use_container_width=True)
                            st.success("Render berhasil! Klik kanan pada gambar untuk menyimpan.")
                            
                    except Exception as e:
                        st.error(f"Gagal merender gambar: {e}")
    else:
        st.caption("Unggah foto dan klik 'Autoscan' di sebelah kiri terlebih dahulu.")
