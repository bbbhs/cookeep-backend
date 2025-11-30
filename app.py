import sqlite3
import pandas as pd
import json
import os
import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
DB_NAME = os.path.join(basedir, 'recipe_recommender.db')
RECIPES_JSON = os.path.join(basedir, 'recipes.json')
MAPPINGS_JSON = os.path.join(basedir, 'mappings.json')
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')

recipes_df = None
material_map = None
material_regex = None

# ------------------------------------------
# 데이터 로드 관련 함수
# ------------------------------------------
def load_data_to_memory():
    global recipes_df, material_map, material_regex

    if recipes_df is not None and material_map is not None:
        return

    if not os.path.exists(DB_NAME):
        initialize_database()
        return

    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        recipes_df = pd.read_sql_query("SELECT * FROM Recipes", conn)

        cursor = conn.cursor()
        cursor.execute("SELECT receipt_item, standard_material FROM MaterialMapping")
        rows = cursor.fetchall()
        conn.close()

        material_map = {item: material for item, material in rows}

        sorted_keys = sorted(material_map.keys(), key=len, reverse=True)
        material_regex = re.compile('|'.join(map(re.escape, sorted_keys)))

        logger.info("데이터 로드 완료")

    except Exception as e:
        logger.error(f"데이터 로드 오류: {e}")

def initialize_database():
    logger.info("DB 초기화 시작")

    if not os.path.exists(RECIPES_JSON) or not os.path.exists(MAPPINGS_JSON):
        logger.error("레시피 JSON 또는 매핑 JSON 없음 → 샘플 생성")
        if not os.path.exists(RECIPES_JSON):
            with open(RECIPES_JSON, 'w', encoding='utf-8') as f:
                json.dump([{
                    "name": "샘플 김치찌개",
                    "materials": {"core": ["김치"], "optional": ["두부"]}
                }], f, ensure_ascii=False, indent=2)

        if not os.path.exists(MAPPINGS_JSON):
            with open(MAPPINGS_JSON, 'w', encoding='utf-8') as f:
                json.dump([{"item": "김치", "material": "김치"}], f, ensure_ascii=False, indent=2)

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS Recipes")
    cur.execute("DROP TABLE IF EXISTS MaterialMapping")

    cur.execute("""
        CREATE TABLE Recipes (
            recipe_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            required_materials TEXT NOT NULL,
            steps TEXT,
            image_url TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE MaterialMapping (
            mapping_id INTEGER PRIMARY KEY,
            receipt_item TEXT NOT NULL UNIQUE,
            standard_material TEXT NOT NULL
        )
    """)

    with open(RECIPES_JSON, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
        for r in recipes:
            cur.execute("INSERT INTO Recipes (name, required_materials, steps, image_url) VALUES (?, ?, ?, ?)",
                        (r["name"], json.dumps(r["materials"], ensure_ascii=False), r.get("steps", ""), r.get("image_url", "")))

    with open(MAPPINGS_JSON, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
        for m in mappings:
            try:
                cur.execute("INSERT INTO MaterialMapping (receipt_item, standard_material) VALUES (?, ?)",
                            (m["item"], m["material"]))
            except:
                pass

    conn.commit()
    conn.close()
    logger.info("DB 초기화 완료")
    load_data_to_memory()

# ------------------------------------------
# 매칭 / 추천 함수
# ------------------------------------------
def calculate_match_score(required, available):
    core = set(required.get('core', []))
    opt = set(required.get('optional', []))
    available = set(available)

    missing_core = core - available
    if missing_core:
        return 0, set(), core.union(opt) - available

    all_required = core.union(opt)
    matched = all_required.intersection(available)
    missing = all_required - available
    ratio = len(matched) / len(all_required) if len(all_required) > 0 else 0

    return ratio, matched, missing

def recommend_recipes(standard_materials, top_n=5):
    global recipes_df
    if recipes_df is None:
        load_data_to_memory()

    recommendations = []
    for _, row in recipes_df.iterrows():
        required = json.loads(row['required_materials'])
        ratio, matched, missing = calculate_match_score(required, standard_materials)
        if ratio > 0:
            recommendations.append({
                "name": row["name"],
                "match_ratio": int(ratio * 100),
                "matched": list(matched),
                "missing": list(missing)
            })

    recommendations.sort(key=lambda x: x["match_ratio"], reverse=True)
    return recommendations[:top_n]

def process_material_lines(lines):
    global material_map, material_regex
    if material_map is None:
        load_data_to_memory()

    std = set()
    for line in lines:
        matches = material_regex.findall(line)
        for m in matches:
            std.add(material_map.get(m))

    return list(std)

# ------------------------------------------
# Flask API
# ------------------------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200

@app.route("/recommend", methods=["POST"])
def recommend_api():
    load_data_to_memory()

    data = request.get_json()
    lines = data.get("receipt_lines", [])

    std = process_material_lines(lines)
    result = recommend_recipes(std, top_n=5)

    return jsonify({
        "status": "success",
        "standard_materials": std,
        "recommendations": result
    })

# ------------------------------------------
# local server
# ------------------------------------------
if __name__ == "__main__":
    load_data_to_memory()
    app.run(host="0.0.0.0", port=5000, debug=True)
