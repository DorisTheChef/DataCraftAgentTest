import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re

# 读取环境变量
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("请先设置 GOOGLE_API_KEY 环境变量。")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

def generate_patients_with_gemini(n=100):
    prompt = (
        f"Generate {n} sets of virtual patient data in JSON array format. "
        "Each patient should have: Name, Age, Gender, and Condition. "
        "Example: "
        '[{"Name": "John Smith", "Age": 45, "Gender": "Male", "Condition": "Diabetes"}, ...]'
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    match = re.search(r'\[.*\]', response.text, re.DOTALL)
    if match:
        patients = json.loads(match.group(0))
        return patients
    else:
        st.error("未能从Gemini返回中提取到JSON数据。")
        return []

st.title("虚拟病人数据生成器 (Gemini AI)")

num = st.number_input("生成病人数量", min_value=1, max_value=200, value=100, step=1)

if st.button("生成数据"):
    with st.spinner("正在生成，请稍候..."):
        patients = generate_patients_with_gemini(num)
        if patients:
            df = pd.DataFrame(patients)
            st.success(f"已生成 {len(df)} 条虚拟病人数据！")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载为CSV文件",
                data=csv,
                file_name="gemini_virtual_patients.csv",
                mime="text/csv"
            )