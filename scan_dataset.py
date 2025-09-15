import os

def scan_dataset(dataset_path):
    """فحص بنية الداتاسيت وعرض التفاصيل"""
    
    if not os.path.exists(dataset_path):
        print(f"❌ المجلد غير موجود: {dataset_path}")
        return
    
    print(f"🔍 فحص المجلد: {dataset_path}")
    print("="*60)
    
    # عد الملفات والمجلدات
    total_files = 0
    total_dirs = 0
    image_files = 0
    
    # أنواع الصور المدعومة
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    print("📁 بنية المجلد:")
    
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root)
        
        if level == 0:
            print(f"{indent}📂 {dataset_path}/")
        else:
            print(f"{indent}📁 {folder_name}/")
        
        # عد المجلدات
        total_dirs += len(dirs)
        
        # عد الملفات والصور
        folder_images = 0
        for file in files:
            total_files += 1
            if file.lower().endswith(image_extensions):
                image_files += 1
                folder_images += 1
        
        # عرض عينة من الملفات
        if files:
            sample_files = files[:3]  # أول 3 ملفات كعينة
            for file in sample_files:
                if file.lower().endswith(image_extensions):
                    print(f"{indent}  🖼️  {file}")
                else:
                    print(f"{indent}  📄 {file}")
            
            if len(files) > 3:
                print(f"{indent}  ... و {len(files)-3} ملف آخر")
        
        if folder_images > 0:
            print(f"{indent}  📊 إجمالي الصور: {folder_images}")
        
        # عدم التعمق أكثر من 3 مستويات
        if level >= 2:
            dirs.clear()
    
    print("\n" + "="*60)
    print("📊 إحصائيات الداتاسيت:")
    print(f"📁 إجمالي المجلدات: {total_dirs}")
    print(f"📄 إجمالي الملفات: {total_files}")
    print(f"🖼️  إجمالي الصور: {image_files}")
    
    # تحليل أسماء الملفات لاستخراج الكلاسات
    print("\n🔍 تحليل أسماء الملفات:")
    sample_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                sample_files.append(file)
                if len(sample_files) >= 10:  # أخذ عينة من 10 ملفات
                    break
        if len(sample_files) >= 10:
            break
    
    print("📋 عينة من أسماء الملفات:")
    for i, filename in enumerate(sample_files, 1):
        print(f"  {i:2d}. {filename}")
    
    # محاولة تحديد نمط التسمية
    print("\n🧠 تحليل نمط التسمية:")
    
    if sample_files:
        first_file = sample_files[0]
        
        if '___' in first_file:
            print("✅ يبدو أن الملفات تتبع نمط PlantVillage: Plant___Disease_number.jpg")
            parts = first_file.split('___')
            print(f"   مثال: {parts[0]} (النبات) ___ {parts[1]} (المرض)")
        
        elif '_' in first_file:
            print("✅ يبدو أن الملفات تحتوي على underscores")
            parts = first_file.split('_')
            print(f"   أجزاء اسم الملف: {parts[:3]}...")  # أول 3 أجزاء
        
        else:
            print("⚠️ لا يوجد نمط واضح في أسماء الملفات")
    
    # اقتراحات للخطوة التالية
    print("\n💡 الخطوات المقترحة:")
    
    if total_dirs > 1:
        print("1️⃣ الداتاسيت يحتوي على مجلدات فرعية")
        print("   → استخدم سكريبت تنظيم المجلدات")
    else:
        print("1️⃣ الداتاسيت يحتوي على ملفات مخلوطة")
        print("   → استخدم سكريبت تنظيم الملفات المخلوطة")
    
    print("2️⃣ بعد التنظيم، ستحتاج تقسيم البيانات لـ train/val")
    print("3️⃣ ثم بناء وتدريب النموذج")
    
    return {
        'total_dirs': total_dirs,
        'total_files': total_files, 
        'image_files': image_files,
        'sample_files': sample_files
    }

if __name__ == "__main__":
    print("🌱 فاحص بنية الداتاسيت")
    print("="*40)
    
    # طلب مسار الداتاسيت
    dataset_path = input("📁 أدخل مسار مجلد الداتاسيت: ").strip()
    
    if not dataset_path:
        # مسارات افتراضية محتملة
        possible_paths = [
            "PlantVillage",
            "plantvillage", 
            "dataset",
            "data",
            "plant_disease_dataset"
        ]
        
        print("🔍 البحث عن مسارات افتراضية...")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ تم العثور على: {path}")
                dataset_path = path
                break
        
        if not dataset_path:
            print("❌ لم يتم العثور على أي مجلد داتاسيت")
            exit()
    
    # فحص الداتاسيت
    result = scan_dataset(dataset_path)
    
    if result:
        print(f"\n🎯 الداتاسيت جاهز للمعالجة!")
        print("📞 شارك النتائج عشان نحدد الخطوة التالية")