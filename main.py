"""Ứng dụng web RAG sử dụng Streamlit, LangChain, Qdrant và Gemini"""
import os
import glob
from typing import List, Optional
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

