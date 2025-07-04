"""
Cloudinary Upload Script for Oil Spill Test Images
Uploads all test images from the dataset to Cloudinary for production use
"""

import os
import cloudinary
import cloudinary.uploader
from pathlib import Path
import json
from tqdm import tqdm

# Cloudinary configuration
def get_cloudinary_credentials():
    """Get Cloudinary credentials from environment variables or user input"""
    cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
    api_key = os.environ.get('CLOUDINARY_API_KEY')
    api_secret = os.environ.get('CLOUDINARY_API_SECRET')
    
    if not all([cloud_name, api_key, api_secret]):
        print("📋 Cloudinary credentials not found in environment variables.")
        print("You can get these from: https://console.cloudinary.com/")
        print("\nPlease enter your Cloudinary credentials:")
        
        if not cloud_name:
            cloud_name = input("Cloud Name: ").strip()
        if not api_key:
            api_key = input("API Key: ").strip()
        if not api_secret:
            api_secret = input("API Secret: ").strip()
    
    return cloud_name, api_key, api_secret

def setup_cloudinary():
    """Configure Cloudinary with your credentials"""
    cloud_name, api_key, api_secret = get_cloudinary_credentials()
    
    if not all([cloud_name, api_key, api_secret]):
        raise ValueError("❌ All Cloudinary credentials are required!")
    
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )
    
    print(f"✅ Cloudinary configured for cloud: {cloud_name}")
    return True

def upload_test_images():
    """Upload all test images to Cloudinary"""
    
    # Path to your test images
    images_path = "backend/notebooks/data/oil-spill/test/images"
    
    if not os.path.exists(images_path):
        print(f"❌ Images path not found: {images_path}")
        return
    
    # Get all jpg files
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith('.jpg')]
    
    print(f"📸 Found {len(image_files)} test images to upload")
    
    uploaded_urls = {}
    failed_uploads = []
    
    # Upload each image with progress bar
    for image_file in tqdm(image_files, desc="Uploading to Cloudinary"):
        try:
            image_path = os.path.join(images_path, image_file)
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                image_path,
                folder="oil-spill-test-images",  # Organize in a folder
                public_id=image_file.replace('.jpg', ''),  # Use filename as public_id
                resource_type="image",
                format="jpg",
                quality="auto",  # Automatic quality optimization
                fetch_format="auto"  # Automatic format optimization
            )
            
            uploaded_urls[image_file] = result['secure_url']
            
        except Exception as e:
            print(f"❌ Failed to upload {image_file}: {str(e)}")
            failed_uploads.append(image_file)
    
    # Save the URLs to a JSON file
    output_file = "cloudinary_test_images.json"
    with open(output_file, 'w') as f:
        json.dump({
            'uploaded_urls': uploaded_urls,
            'failed_uploads': failed_uploads,
            'total_uploaded': len(uploaded_urls),
            'total_failed': len(failed_uploads)
        }, f, indent=2)
    
    print(f"\n🎉 Upload Complete!")
    print(f"✅ Successfully uploaded: {len(uploaded_urls)} images")
    print(f"❌ Failed uploads: {len(failed_uploads)} images")
    print(f"📄 URLs saved to: {output_file}")
    
    return uploaded_urls

def generate_test_images_config(uploaded_urls):
    """Generate the test-images.ts file with Cloudinary URLs"""
    
    # Create categories based on image numbers for variety
    categories = ['satellite', 'coastal', 'offshore', 'complex']
    difficulties = ['easy', 'medium', 'hard']
    
    test_images = []
    
    for i, (filename, url) in enumerate(uploaded_urls.items()):
        # Extract image number from filename (img_0001.jpg -> 1)
        img_num = int(filename.replace('img_', '').replace('.jpg', ''))
        
        # Assign category and difficulty based on image number
        category = categories[img_num % len(categories)]
        difficulty = difficulties[img_num % len(difficulties)]
        
        test_images.append({
            'id': img_num,
            'name': f"Test Image {img_num:03d}",
            'description': f"SAR satellite image - Oil spill detection test case {img_num}",
            'url': url,
            'category': category,
            'difficulty': difficulty,
            'expectedResult': "Oil Spill Analysis"
        })
    
    # Generate TypeScript file content
    ts_content = f"""// Test images data - hosted on Cloudinary for production
// Total images available: {len(test_images)}
export const ALL_TEST_IMAGES = {json.dumps(test_images, indent=2)}

// Randomly select images for display
export const getRandomTestImages = (count: number = 20) => {{
  const shuffled = [...ALL_TEST_IMAGES].sort(() => 0.5 - Math.random())
  return shuffled.slice(0, count)
}}

// Get images by category
export const getImagesByCategory = (category: string) => {{
  if (category === 'all') return ALL_TEST_IMAGES
  return ALL_TEST_IMAGES.filter(img => img.category === category)
}}

// Get images by difficulty
export const getImagesByDifficulty = (difficulty: string) => {{
  if (difficulty === 'all') return ALL_TEST_IMAGES
  return ALL_TEST_IMAGES.filter(img => img.difficulty === difficulty)
}}

// Default test images (random 20)
export const TEST_IMAGES = getRandomTestImages(20)

// Category colors for UI
export const DIFFICULTY_COLORS = {{
  easy: "bg-green-100 text-green-800 border-green-200",
  medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  hard: "bg-red-100 text-red-800 border-red-200"
}}

// Category icons mapping
export const CATEGORY_ICONS = {{
  "satellite": "🛰️",
  "coastal": "🏖️",
  "offshore": "🌊",
  "complex": "🌀",
  "test-data": "📊"
}}
"""
    
    # Save the TypeScript file
    with open("src/data/test-images-cloudinary.ts", 'w', encoding='utf-8') as f:
        f.write(ts_content)
    
    print(f"📝 Generated TypeScript config: src/data/test-images-cloudinary.ts")

if __name__ == "__main__":
    print("🚀 Starting Cloudinary Upload Process...")
    print("   This will upload all test images to Cloudinary for production use\n")
    
    try:
        # Setup Cloudinary
        setup_cloudinary()
        
        # Upload images
        uploaded_urls = upload_test_images()
        
        if uploaded_urls:
            # Generate TypeScript config
            generate_test_images_config(uploaded_urls)
            print("\n🎉 All done! Your test images are now hosted on Cloudinary!")
            print("   You can now use the generated 'test-images-cloudinary.ts' file in production")
        else:
            print("❌ No images were uploaded. Please check your configuration.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your Cloudinary credentials and try again.")
