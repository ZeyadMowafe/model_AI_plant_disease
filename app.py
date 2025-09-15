from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import logging

# إعداد التطبيق
app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant_disease_detection_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# إعداد المجلدات
UPLOAD_FOLDER = "static/uploads"
MODELS_FOLDER = "models"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# إنشاء المجلدات المطلوبة
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# متغيرات عامة
model = None
class_names = []
model_info = {}

def load_model_and_classes():
    """تحميل النموذج وأسماء الكلاسات"""
    global model, class_names, model_info
    
    model_path = os.path.join(MODELS_FOLDER, 'plant_disease_cnn.h5')
    best_model_path = os.path.join(MODELS_FOLDER, 'best_model.h5')
    class_names_path = os.path.join(MODELS_FOLDER, 'class_names.txt')
    info_path = os.path.join(MODELS_FOLDER, 'training_info.json')
    
    try:
        # تجربة تحميل أفضل نموذج أولاً
        if os.path.exists(best_model_path):
            model = tf.keras.models.load_model(best_model_path)
            logger.info("✅ تم تحميل أفضل نموذج")
        elif os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("✅ تم تحميل النموذج الرئيسي")
        else:
            logger.error("❌ لم يتم العثور على النموذج!")
            return False
        
        # تحميل أسماء الكلاسات
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
            logger.info(f"✅ تم تحميل {len(class_names)} كلاس")
        else:
            # كلاسات افتراضية إذا لم يكن الملف موجود
            class_names = [
                'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
                'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus'
            ]
            logger.warning("⚠️ استخدام كلاسات افتراضية")
        
        # تحميل معلومات التدريب
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            logger.info("✅ تم تحميل معلومات التدريب")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل النموذج: {e}")
        return False

def allowed_file(filename):
    """التحقق من نوع الملف المسموح"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_class_name(class_name):
    """استخراج معلومات النبات والمرض من اسم الكلاس"""
    try:
        if '___' in class_name:
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ').title()
            disease = parts[1].replace('_', ' ').title()
        elif '_' in class_name:
            parts = class_name.split('_')
            plant = parts[0].title()
            disease = ' '.join(parts[1:]).replace('_', ' ').title()
        else:
            plant = class_name.title()
            disease = "Unknown"
        
        return plant, disease
    except:
        return "Unknown Plant", "Unknown Disease"

def get_treatment_advice(plant, disease):
    """إعطاء نصائح العلاج حسب النبات والمرض"""
    treatments = {
        # أمراض الطماطم
        'bacterial spot': 'استخدم مبيدات بكتيرية نحاسية وتحسين التهوية وتجنب الري على الأوراق',
        'early blight': 'رش بمبيدات فطرية مثل الكلوروثالونيل وإزالة الأوراق المصابة',
        'late blight': 'استخدم مبيدات فطرية وقائية وتحسين التهوية وتقليل الرطوبة',
        'leaf mold': 'تحسين التهوية وتقليل الرطوبة واستخدام مبيدات فطرية',
        'septoria leaf spot': 'إزالة الأوراق المصابة واستخدام مبيدات فطرية وتجنب الري العلوي',
        'spider mites': 'استخدام مبيدات العناكب واستخدام الماء لغسل الأوراق',
        'target spot': 'مبيدات فطرية وتحسين دوران الهواء حول النباتات',
        'mosaic virus': 'إزالة النباتات المصابة ومكافحة الحشرات الناقلة للفيروس',
        'yellow leaf curl virus': 'مكافحة الذبابة البيضاء وإزالة النباتات المصابة',
        
        # أمراض البطاطس
        'early blight': 'رش وقائي بمبيدات فطرية وتحسين التهوية',
        'late blight': 'مبيدات فطرية وقائية منتظمة وتجنب الري المفرط',
        
        # أمراض الفلفل
        'bacterial spot': 'مبيدات بكتيرية نحاسية وتحسين الصرف',
        
        # النباتات الصحية
        'healthy': 'النبات صحي! حافظ على الري المنتظم والتسميد المتوازن'
    }
    
    disease_lower = disease.lower()
    for key, treatment in treatments.items():
        if key in disease_lower:
            return f"🌿 علاج {plant}: {treatment}"
    
    return f"استشر خبير زراعي لعلاج {disease} في {plant}"

def predict_disease(img_path):
    """التنبؤ بمرض النبات من الصورة"""
    if model is None:
        return {"error": "النموذج غير محمل"}, 0.0
    
    try:
        # تحديد حجم الصورة من معلومات النموذج
        img_size = (224, 224)  # default
        if 'img_width' in model_info and 'img_height' in model_info:
            img_size = (model_info['img_width'], model_info['img_height'])
        
        # تحميل وتحضير الصورة
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # التنبؤ
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        # التأكد من صحة الفهرس
        if class_idx >= len(class_names):
            return {"error": "خطأ في فهرس الكلاس"}, 0.0
        
        predicted_class = class_names[class_idx]
        plant, disease = parse_class_name(predicted_class)
        treatment = get_treatment_advice(plant, disease)
        
        # تحديد مستوى الثقة
        confidence_level = "عالي" if confidence > 0.8 else "متوسط" if confidence > 0.6 else "منخفض"
        
        result = {
            'plant': plant,
            'disease': disease,
            'full_prediction': predicted_class.replace('_', ' '),
            'confidence_level': confidence_level,
            'treatment': treatment
        }
        
        return result, float(confidence)
        
    except Exception as e:
        logger.error(f"خطأ في التنبؤ: {e}")
        return {"error": f"خطأ في التنبؤ: {str(e)}"}, 0.0

def initialize_app():
    """تشغيل المتطلبات الأولية"""
    logger.info("🚀 بدء تطبيق Plant Disease Detection")
    
    if not load_model_and_classes():
        logger.warning("⚠️ لم يتم تحميل النموذج. تأكد من وجود ملفات النموذج في مجلد models/")
        logger.info("💡 لتدريب النموذج، شغل: python train_model.py")

@app.route("/")
def index():
    """الصفحة الرئيسية"""
    return render_template("index.html", 
                         model_loaded=model is not None,
                         num_classes=len(class_names),
                         model_info=model_info)

@app.route("/upload", methods=["POST"])
def upload_file():
    """رفع الملف والتنبؤ"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if file and allowed_file(file.filename):
            # حفظ الملف
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # تسجيل معلومات الملف
            file_size = os.path.getsize(filepath)
            logger.info(f"تم رفع ملف: {filename}, الحجم: {file_size} bytes")

            # التنبؤ
            prediction, confidence = predict_disease(filepath)
            
            if 'error' in prediction:
                return jsonify({'error': prediction['error']}), 500
            
            # إعداد الاستجابة
            response_data = {
                'success': True,
                'filename': filename,
                'file_url': f'/static/uploads/{filename}',
                'plant': prediction['plant'],
                'disease': prediction['disease'],
                'full_prediction': prediction['full_prediction'],
                'confidence': f"{confidence:.2%}",
                'confidence_level': prediction['confidence_level'],
                'treatment': prediction['treatment'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # حفظ سجل التنبؤ
            log_prediction(filename, prediction, confidence)
            
            return jsonify(response_data)
        else:
            return jsonify({'error': 'نوع الملف غير مدعوم. استخدم PNG, JPG, JPEG, GIF'}), 400
            
    except Exception as e:
        logger.error(f"خطأ في رفع الملف: {e}")
        return jsonify({'error': f'خطأ في الخادم: {str(e)}'}), 500

def log_prediction(filename, prediction, confidence):
    """تسجيل التنبؤات في ملف سجل"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'prediction': prediction,
            'confidence': float(confidence)
        }
        
        log_file = 'models/predictions_log.json'
        
        # قراءة السجل الموجود
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # إضافة التنبؤ الجديد
        logs.append(log_entry)
        
        # الاحتفاظ بآخر 1000 تنبؤ فقط
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # حفظ السجل
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"خطأ في حفظ السجل: {e}")

@app.route("/health")
def health_check():
    """فحص صحة التطبيق"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat()
    })

@app.route("/model-info")
def model_info_endpoint():
    """معلومات النموذج"""
    return jsonify({
        'model_loaded': model is not None,
        'num_classes': len(class_names),
        'class_names': class_names[:10],  # أول 10 كلاسات
        'model_info': model_info,
        'available_files': {
            'model': os.path.exists(os.path.join(MODELS_FOLDER, 'plant_disease_cnn.h5')),
            'best_model': os.path.exists(os.path.join(MODELS_FOLDER, 'best_model.h5')),
            'class_names': os.path.exists(os.path.join(MODELS_FOLDER, 'class_names.txt')),
            'training_info': os.path.exists(os.path.join(MODELS_FOLDER, 'training_info.json'))
        }
    })

@app.route("/stats")
def get_stats():
    """إحصائيات الاستخدام"""
    try:
        log_file = 'models/predictions_log.json'
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # إحصائيات بسيطة
            total_predictions = len(logs)
            recent_predictions = [log for log in logs 
                                if datetime.fromisoformat(log['timestamp']) > 
                                   datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]
            
            # أكثر الأمراض تشخيصاً
            disease_counts = {}
            for log in logs:
                disease = log['prediction'].get('disease', 'Unknown')
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return jsonify({
                'total_predictions': total_predictions,
                'today_predictions': len(recent_predictions),
                'top_diseases': top_diseases,
                'avg_confidence': np.mean([log['confidence'] for log in logs]) if logs else 0
            })
        else:
            return jsonify({
                'total_predictions': 0,
                'today_predictions': 0,
                'top_diseases': [],
                'avg_confidence': 0
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """معالجة الملفات الكبيرة"""
    return jsonify({'error': 'الملف كبير جداً. الحد الأقصى 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """معالجة الصفحات غير الموجودة"""
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    """معالجة الأخطاء الداخلية"""
    return jsonify({'error': 'خطأ في الخادم'}), 500

if __name__ == "__main__":
    print("🌱 Plant Disease Detection App")
    print("="*50)
    
    # تهيئة التطبيق
    initialize_app()
    
    # تحميل النموذج
    if model is None:
        print("⚠️ تحذير: لم يتم تحميل النموذج")
        print("💡 لتدريب النموذج أولاً، شغل: python train_model.py")
        print("📝 أو ضع ملفات النموذج في مجلد models/")
    else:
        print(f"✅ تم تحميل النموذج مع {len(class_names)} كلاس")
    
    print("\n🚀 بدء الخادم...")
    print("🌐 افتح المتصفح على: http://localhost:5000")
    print("⏹️ اضغط Ctrl+C لإيقاف الخادم")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n✅ تم إيقاف الخادم")
    except Exception as e:
        print(f"\n❌ خطأ في تشغيل الخادم: {e}")