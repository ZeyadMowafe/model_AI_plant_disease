import os
import shutil
import random
from collections import defaultdict

def clean_and_organize_dataset(source_dir, output_dir="dataset", train_ratio=0.8):
    """
    تنظيف البيانات المتكررة وتقسيمها إلى train/val
    """
    print("🧹 بدء تنظيف وتقسيم البيانات...")
    print("="*60)
    
    # إنشاء مجلدات الإخراج
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # مسح المجلدات الموجودة وإعادة إنشائها
    if os.path.exists(output_dir):
        print(f"🗑️ مسح المجلد الموجود: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # جمع جميع الكلاسات (تجنب التكرار)
    class_data = defaultdict(set)  # استخدام set لتجنب التكرار
    
    print(f"🔍 فحص المجلد: {source_dir}")
    
    # البحث في جميع المجلدات الفرعية
    for root, dirs, files in os.walk(source_dir):
        folder_name = os.path.basename(root)
        
        # تجاهل المجلدات الفرعية العميقة
        level = root.replace(source_dir, '').count(os.sep)
        if level > 2:
            continue
            
        # التحقق من أن هذا مجلد كلاس (يحتوي على صور)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files and folder_name != os.path.basename(source_dir):
            # تنظيف اسم الكلاس
            class_name = clean_class_name(folder_name)
            
            # إضافة الصور للكلاس (set تمنع التكرار)
            for image_file in image_files:
                image_path = os.path.join(root, image_file)
                # استخدام اسم الملف كمعرف فريد لتجنب التكرار
                class_data[class_name].add((image_path, image_file))
    
    print(f"📊 تم العثور على {len(class_data)} كلاس:")
    
    total_train_images = 0
    total_val_images = 0
    
    # معالجة كل كلاس
    for class_name, image_set in class_data.items():
        images_list = list(image_set)  # تحويل set إلى list
        num_images = len(images_list)
        
        print(f"  📂 {class_name}: {num_images} صورة")
        
        # إنشاء مجلدات الكلاس
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # خلط الصور عشوائياً
        random.shuffle(images_list)
        
        # تحديد نقطة التقسيم
        split_point = int(num_images * train_ratio)
        
        # نسخ صور التدريب
        train_images = images_list[:split_point]
        for i, (image_path, original_name) in enumerate(train_images):
            # إنشاء اسم ملف جديد منظم
            extension = os.path.splitext(original_name)[1]
            new_name = f"{class_name}_{i:04d}{extension}"
            dst_path = os.path.join(train_class_dir, new_name)
            
            try:
                shutil.copy2(image_path, dst_path)
            except Exception as e:
                print(f"    ⚠️ خطأ في نسخ {original_name}: {e}")
        
        # نسخ صور التقييم  
        val_images = images_list[split_point:]
        for i, (image_path, original_name) in enumerate(val_images):
            extension = os.path.splitext(original_name)[1]
            new_name = f"{class_name}_val_{i:04d}{extension}"
            dst_path = os.path.join(val_class_dir, new_name)
            
            try:
                shutil.copy2(image_path, dst_path)
            except Exception as e:
                print(f"    ⚠️ خطأ في نسخ {original_name}: {e}")
        
        train_count = len(train_images)
        val_count = len(val_images)
        
        total_train_images += train_count
        total_val_images += val_count
        
        print(f"    ✅ تدريب: {train_count}, تقييم: {val_count}")
    
    print(f"\n🎉 تم التنظيف والتقسيم بنجاح!")
    print(f"📈 إجمالي صور التدريب: {total_train_images}")
    print(f"📊 إجمالي صور التقييم: {total_val_images}")
    print(f"📁 مجلد الإخراج: {output_dir}")
    
    # حفظ معلومات الكلاسات
    save_class_info(class_data.keys(), output_dir, total_train_images + total_val_images)
    
    return True

def clean_class_name(folder_name):
    """
    تنظيف أسماء الكلاسات لتكون موحدة
    """
    # إزالة المسافات الزائدة والرموز الغريبة
    class_name = folder_name.strip()
    
    # استبدال الرموز الخاصة
    replacements = {
        '__': '___',  # توحيد الفواصل
        ' ': '_',     # استبدال المسافات
        '(': '',      # إزالة الأقواس
        ')': '',
        ',': '',      # إزالة الفواصل
    }
    
    for old, new in replacements.items():
        class_name = class_name.replace(old, new)
    
    return class_name

def save_class_info(class_names, output_dir, total_images):
    """
    حفظ معلومات الكلاسات في ملف
    """
    info_file = os.path.join(output_dir, 'dataset_info.txt')
    
    sorted_classes = sorted(class_names)
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("Plant Disease Dataset Information\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Classes: {len(sorted_classes)}\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Train/Val Split: 80%/20%\n\n")
        f.write("Classes List:\n")
        f.write("-"*30 + "\n")
        
        for i, class_name in enumerate(sorted_classes):
            plant, disease = parse_class_name(class_name)
            f.write(f"{i:2d}. {class_name}\n")
            f.write(f"    Plant: {plant}\n")
            f.write(f"    Disease: {disease}\n\n")
    
    # حفظ قائمة بسيطة للاستخدام في الكود
    classes_file = os.path.join(output_dir, 'class_names.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in sorted_classes:
            f.write(f"{class_name}\n")
    
    print(f"💾 تم حفظ معلومات الداتاسيت في:")
    print(f"    📄 {info_file}")
    print(f"    📄 {classes_file}")

def parse_class_name(class_name):
    """
    استخراج اسم النبات والمرض من اسم الكلاس
    """
    if '___' in class_name:
        parts = class_name.split('___')
        plant = parts[0].replace('_', ' ').title()
        disease = parts[1].replace('_', ' ').title()
    else:
        parts = class_name.split('_')
        plant = parts[0].title()
        disease = ' '.join(parts[1:]).title() if len(parts) > 1 else 'Unknown'
    
    return plant, disease

def verify_dataset(dataset_dir):
    """
    التحقق من صحة التقسيم
    """
    print(f"\n🔍 التحقق من صحة التقسيم في: {dataset_dir}")
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("❌ مجلدات train أو val غير موجودة!")
        return False
    
    train_classes = set(os.listdir(train_dir))
    val_classes = set(os.listdir(val_dir))
    
    print(f"📂 كلاسات التدريب: {len(train_classes)}")
    print(f"📂 كلاسات التقييم: {len(val_classes)}")
    
    # التحقق من تطابق الكلاسات
    missing_in_val = train_classes - val_classes
    missing_in_train = val_classes - train_classes
    
    if missing_in_val:
        print(f"⚠️ كلاسات مفقودة في val: {missing_in_val}")
    
    if missing_in_train:
        print(f"⚠️ كلاسات مفقودة في train: {missing_in_train}")
    
    if not missing_in_val and not missing_in_train:
        print("✅ جميع الكلاسات متطابقة في train و val")
        
        # عد الصور في كل مجلد
        total_train = 0
        total_val = 0
        
        for class_name in train_classes:
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            
            train_count = len([f for f in os.listdir(train_class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            val_count = len([f for f in os.listdir(val_class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            total_train += train_count
            total_val += val_count
        
        print(f"📈 إجمالي صور التدريب: {total_train}")
        print(f"📊 إجمالي صور التقييم: {total_val}")
        print(f"📏 نسبة التقسيم: {total_train/(total_train+total_val)*100:.1f}% / {total_val/(total_train+total_val)*100:.1f}%")
        
        return True
    
    return False

if __name__ == "__main__":
    print("🌱 تنظيف وتقسيم بيانات Plant Disease")
    print("="*50)
    
    # مسار البيانات الأصلي
    source_path = r"data\dataset\PlantVillage"
    output_path = "dataset"
    
    # التأكد من وجود المسار
    if not os.path.exists(source_path):
        print(f"❌ المسار غير موجود: {source_path}")
        source_path = input("أدخل المسار الصحيح: ").strip()
        
        if not os.path.exists(source_path):
            print("❌ المسار غير صحيح!")
            exit()
    
    print(f"📁 المسار المصدر: {source_path}")
    print(f"📁 مسار الإخراج: {output_path}")
    
    # البدء في المعالجة
    success = clean_and_organize_dataset(source_path, output_path)
    
    if success:
        # التحقق من النتيجة
        verify_dataset(output_path)
        
        print(f"\n🚀 الخطوات التالية:")
        print("1️⃣ تشغيل: python train_model.py")
        print("2️⃣ تشغيل: python app.py") 
        print("3️⃣ فتح: http://localhost:5000")
    else:
        print("❌ فشل في تنظيم البيانات!")