import sqlite3
import pandas as pd
import json
import os
import io
import re 
import logging
import urllib.parse 
import requests 

# 💡 Render 인증을 위해 google-auth 모듈 import
import google.oauth2.service_account 
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from google.cloud import vision
from flask_cors import CORS 

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 📌 1. 데이터베이스 및 전역 변수 설정 (초기값만 설정)
# ----------------------------------------------------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
DB_NAME = os.path.join(basedir, 'recipe_recommender.db')
RECIPES_JSON = os.path.join(basedir, 'recipes.json')
MAPPINGS_JSON = os.path.join(basedir, 'mappings.json')
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')

recipes_df = None       # 💡 None으로 변경: 필요할 때 로드
material_map = None     # 💡 None으로 변경: 필요할 때 로드
material_regex = None
vision_client = None    # 💡 None으로 유지

# ----------------------------------------------------------------------
# 📌 2. 데이터 및 로직 함수 (일부 수정)
# ----------------------------------------------------------------------
def load_data_to_memory():
    """DB의 모든 데이터를 메모리(전역 변수)로 로드합니다."""
    # 💡 [핵심 수정] 함수 내에서만 전역 변수를 사용하도록 global 선언
    global recipes_df, material_map, material_regex
    
    if recipes_df is not None and material_map is not None:
        return # 이미 로드됨

    if not os.path.exists(DB_NAME):
        logger.info("DB 파일이 존재하지 않아 새로 생성합니다.")
        initialize_database() 
        # initialize_database() 안에서 다시 load_data_to_memory()를 호출함
        return
    
    try:
        # DB 로드 로직 (생략 - 변경 없음)
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        recipes_df = pd.read_sql_query("SELECT * FROM Recipes", conn)
        cursor = conn.cursor()
        cursor.execute("SELECT receipt_item, standard_material FROM MaterialMapping")
        rows = cursor.fetchall()
        conn.close()
        material_map = {item: material for item, material in rows}
        if recipes_df.empty or not material_map:
             logger.warning("❌ [경고] 데이터(레시피 또는 매핑)가 비어있습니다. DB를 초기화합니다.")
             initialize_database()
        else:
            sorted_keys = sorted(material_map.keys(), key=len, reverse=True)
            regex_pattern = '|'.join(re.escape(key) for key in sorted_keys)
            material_regex = re.compile(regex_pattern)
            logger.info(f"✅ 매핑 데이터 {len(material_map)}건 메모리 로드 완료.")
    except Exception as e:
        logger.error(f"❌ 데이터 로드 중 오류: {e}")

def initialize_database():
    # ... (DB 초기화 로직 - 변경 없음)
    logger.info("⏳ 데이터베이스 초기화를 시작합니다...")
    if not os.path.exists(RECIPES_JSON) or not os.path.exists(MAPPINGS_JSON):
        logger.error(f"❌ [오류] {RECIPES_JSON} 또는 {MAPPINGS_JSON} 파일이 없습니다.")
        # 샘플 파일 생성 로직 (생략)
        if not os.path.exists(RECIPES_JSON):
             with open(RECIPES_JSON, 'w', encoding='utf-8') as f: json.dump([{"name": "샘플 김치찌개", "materials": {"core": ["김치"], "optional": ["두부"]}}], f, ensure_ascii=False, indent=2)
        if not os.path.exists(MAPPINGS_JSON):
             with open(MAPPINGS_JSON, 'w', encoding='utf-8') as f: json.dump([{"item": "샘플김치", "material": "김치"}], f, ensure_ascii=False, indent=2)
        return
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS Recipes')
    cursor.execute('DROP TABLE IF EXISTS MaterialMapping')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Recipes (recipe_id INTEGER PRIMARY KEY, name TEXT NOT NULL, required_materials TEXT NOT NULL, steps TEXT, image_url TEXT)
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS MaterialMapping (mapping_id INTEGER PRIMARY KEY, receipt_item TEXT NOT NULL UNIQUE, standard_material TEXT NOT NULL)
    ''')
    try:
        with open(RECIPES_JSON, 'r', encoding='utf-8') as f: sample_recipes = json.load(f)
        for recipe in sample_recipes:
            steps = recipe.get("steps", "")
            image_url = recipe.get("image_url", "default_image_url")
            cursor.execute('INSERT INTO Recipes (name, required_materials, steps, image_url) VALUES (?, ?, ?, ?)', (recipe['name'], json.dumps(recipe['materials'], ensure_ascii=False), steps, image_url))
        with open(MAPPINGS_JSON, 'r', encoding='utf-8') as f: sample_mappings = json.load(f)
        for mapping in sample_mappings:
            try:
                cursor.execute('INSERT INTO MaterialMapping (receipt_item, standard_material) VALUES (?, ?)', (mapping['item'], mapping['material']))
            except sqlite3.IntegrityError: pass
        conn.commit()
    except Exception as e:
        logger.error(f"❌ DB 삽입 중 오류: {e}")
    finally:
        conn.close()
    logger.info("✅ 데이터베이스 초기화 완료. 데이터 메모리 로드를 다시 시도합니다.")
    load_data_to_memory()

def calculate_match_score(required_data, available_materials):
    # ... (calculate_match_score 함수 내용 - 변경 없음)
    required_core = set(required_data.get('core', [])) if isinstance(required_data, dict) else set(required_data)
    required_optional = set(required_data.get('optional', [])) if isinstance(required_data, dict) else set()
    available_set = set(available_materials)
    if not required_core and not required_optional: return 0.0, set(), set()
    if not required_core and required_optional: required_core = required_optional; required_optional = set()
    missing_core = required_core.difference(available_set)
    if len(missing_core) > 0:
        all_required = required_core.union(required_optional)
        matched = all_required.intersection(available_set)
        missing = all_required.difference(available_set)
        return 0.0, matched, missing 
    all_required = required_core.union(required_optional)
    matched = all_required.intersection(available_set)
    missing = all_required.difference(available_set)
    match_ratio = len(matched) / len(all_required) if len(all_required) > 0 else 0.0
    return match_ratio, matched, missing

def recommend_recipes(standard_materials, top_n=5):
    # ... (recommend_recipes 함수 내용 - 변경 없음)
    global recipes_df
    if recipes_df is None or recipes_df.empty:
        load_data_to_memory()
        if recipes_df is None or recipes_df.empty:
             logger.error("추천 로직 실행 중... 레시피 데이터 로드 실패.")
             return []

    recommendations = []
    for _, row in recipes_df.iterrows():
        try:
            required_data_obj = json.loads(row['required_materials'])
            ratio, matched, missing = calculate_match_score(required_data_obj, standard_materials)
            if ratio > 0:
                recommendations.append({
                    'name': row['name'], 'image_url': row['image_url'],
                    'match_ratio': int(ratio * 100),
                    'matched_materials': list(matched), 'missing_materials': list(missing),
                    'missing_count': len(missing), 'steps': row['steps']
                })
        except Exception as e:
            logger.warning(f"레시피 '{row['name']}' 처리 중 오류: {e}")
    recommendations.sort(key=lambda x: (x['match_ratio'], -x['missing_count']), reverse=True)
    return recommendations[:top_n]

def process_receipt_to_recommend(receipt_lines):
    # ... (process_receipt_to_recommend 함수 내용 - 변경 없음)
    global material_map, material_regex
    
    # 💡 [핵심 수정] 데이터가 없으면 로드 시도
    if material_map is None:
        load_data_to_memory()
        if material_map is None:
             logger.error("매칭 로직 실행 중... 매핑 데이터 로드 실패.")
             return []
    
    standard_materials = set()
    for line in receipt_lines:
        cleaned_line = line.strip()
        if not cleaned_line: continue
        matches = material_regex.findall(cleaned_line)
        if matches:
            for matched_key in matches:
                standard_material = material_map.get(matched_key)
                if standard_material:
                    standard_materials.add(standard_material)
                    
    logger.info(f"정규화된 재료: {list(standard_materials)}")
    return recommend_recipes(list(standard_materials), top_n=5)

# ----------------------------------------------------------------------
# 📌 3. API 설정 초기화 함수 (문제 해결)
# ----------------------------------------------------------------------
def _init_vision_client():
    """vision_client 전역 변수를 초기화하는 함수"""
    global vision_client
    
    # 💡 이미 초기화 되었다면 재실행 방지
    if vision_client is not None:
        return vision_client

    try:
        # 1. Render 환경 변수에서 키를 읽어옴
        json_key_text = os.environ.get('KEY_FILE_JSON')
        
        if json_key_text:
            # Render: JSON 텍스트를 파싱하고 메모리에서 인증
            credentials_info = json.loads(json_key_text)
            credentials = google.oauth2.service_account.Credentials.from_service_account_info(credentials_info)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            logger.info("✅ Google Vision 클라이언트 (Render Env) 초기화 성공.")

        else:
            # 2. 로컬 테스트용: my-key.json 파일을 읽음
            credential_path = os.path.join(basedir, 'my-key.json')
            if not os.path.exists(credential_path):
                raise FileNotFoundError(f"'{credential_path}' 파일을 찾을 수 없습니다. (Google Vision 기능 불가)")
            
            # 로컬: os.environ에 경로를 설정하고 VisionClient()가 자동으로 찾음
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path 
            vision_client = vision.ImageAnnotatorClient()
            logger.info("✅ Google Vision 클라이언트 (로컬 파일) 초기화 성공.")


    except Exception as e:
        logger.error(f"❌ Google Vision 초기화 실패: {e}")
        vision_client = None
        
    return vision_client


# 💡 모듈 로드 시 바로 호출하지 않고, 라우트에서 호출되도록 변경
# _init_vision_client() 
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 📌 4. Flask API 서버 설정 및 라우트
# ----------------------------------------------------------------------
app = Flask(__name__)
CORS(app) 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 서버 상태 확인용 기본 라우트 (Render 안정화용)
@app.route('/', methods=['GET'])
def home_check():
    """Render가 서버 상태를 확인하기 위한 기본 경로입니다."""
    # 💡 [핵심 수정] 데이터 및 클라이언트를 최초 요청 시에만 로드
    load_data_to_memory()
    _init_vision_client()
    
    return jsonify({'status': 'ok', 'message': 'Recipe Recommender Service is running'}), 200


@app.route('/recommend', methods=['POST'])
def recommend_from_image():
    # 💡 [핵심 수정] 모든 요청 시 데이터/클라이언트 로드 확인 (이미 로드되었으면 바로 통과)
    load_data_to_memory()
    _init_vision_client()
    
    global vision_client
    if vision_client is None:
        logger.error("Vision 클라이언트 초기화 실패로 요청 거부.")
        return jsonify({'status': 'error', 'message': "서버의 Google Vision API 인증에 실패했습니다. (키 파일 확인 필요)"}), 400

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': '이미지 파일이 없습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '선택된 파일이 없습니다.'}), 400

    if file:
        content = file.read()
        
        try:
            # 3. Google Cloud Vision API로 OCR 실행
            image = vision.Image(content=content)
            response = vision_client.text_detection(image=image)
            texts = response.text_annotations

            if response.error.message:
                raise Exception(f'Google Vision API 오류: {response.error.message}')

            full_text = texts[0].description if texts else ""

            # 4. 텍스트를 리스트로 변환 및 처리
            receipt_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            if not receipt_lines:
                logger.warning("OCR 결과가 비어있습니다.")
                return jsonify({'status': 'error', 'message': 'Google OCR 결과가 비어있습니다.'}), 400

            logger.info("--- Google Vision OCR 결과 ---")
            logger.info(full_text)
            logger.info("-----------------------------")

            # 5. 추천 로직 실행
            recommendations = process_receipt_to_recommend(receipt_lines)
            
            # 6. JSON으로 결과 반환
            return jsonify({
                'status': 'success',
                'ocr_lines': receipt_lines,
                'recommendations': recommendations
            })

        except Exception as e:
            logger.error(f"❌ 서버 처리 중 오류: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': f"서버 내부 처리 오류: {e}"}), 500


# ----------------------------------------------------------------------
# 📌 5. 서버 실행
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 로컬에서 실행할 때만 데이터 로드
    load_data_to_memory()
    _init_vision_client() # 로컬에서 Vision 초기화
    app.run(debug=True, host='0.0.0.0', port=5000)