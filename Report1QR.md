# Report 1 QR Code Addition

## Context

**Date**: 2025-10-31
**Task**: Implementing Stage 8 extraction scripts
**File**: `extract_client_data.py` (Report 1 - Client Executive Report)

## Problem

Report 1 (Section 3.2 of Stage8MVP2.md) does not generate QR codes.

**Current state**:
- Report 2 (Creator): ✅ 6 QR codes (2 per bucket × 3 buckets)
- Report 3 (Competitor): ✅ 1 QR code (top performer)
- Report 4 (Multi-Competitor): ✅ N QR codes (1 per competitor)
- Report 1 (Client): ❌ No QR codes

**Current output**:
```
/data/clients/{client}/hashtags/{hashtag}/top_contrastive/
└── {hashtag}_client_data.xlsx
```

**Gap**: Client executives need visual examples to see what top-performing content looks like in each winning bucket.

## Solution

Add 3 QR codes to Report 1: **1 top performer per winning bucket**.

**Selection criteria**:
- Highest engagement rate (calculated via `calculate_engagement_metrics()`)
- One per winning bucket (e.g., 60-90s, 18-33s, 33-60s)

**New output structure**:
```
/data/clients/{client}/hashtags/{hashtag}/top_contrastive/
├── {hashtag}_client_data.xlsx
└── qr_codes/
    ├── {hashtag}_60-90s_top.png
    ├── {hashtag}_18-33s_top.png
    └── {hashtag}_33-60s_top.png
```

---

## Implementation

**File**: `/home/jorge/rumiaifinal/extract_client_data.py`

### **Step 1: Add QR Code Selection and Generation**

**Location**: After STEP 8 (creative formulas), before STEP 9 (Excel structure building)

Insert new section (around line 365):

```python
# =============================
# STEP 8.5: Generate QR Codes (3 total: 1 per winning bucket)
# =============================
print(f"\n📱 Generating QR codes...")

qr_output_dir = os.path.join(base_path, 'qr_codes')
os.makedirs(qr_output_dir, exist_ok=True)

qr_data_list = []
qr_metadata = {}  # Store metadata for Excel

for bucket in winning_buckets:
    bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket}')

    # Select top performer from this bucket
    selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')
    with open(selected_videos_path, 'r') as f:
        data = json.load(f)

    top_count = data['top_count']
    top_videos = data['videos'][:top_count]

    # Get #1 top performer (highest engagement)
    best_video = None
    best_engagement = 0

    for video in top_videos:
        engagement = calculate_engagement_metrics(video)
        if engagement > best_engagement:
            best_engagement = engagement
            best_video = video

    if best_video:
        # Add to QR generation list
        qr_data_list.append({
            'filename': f"{args.hashtag}_{bucket}_top.png",
            'url': best_video['webVideoUrl']
        })

        # Store metadata for Excel
        qr_metadata[bucket] = {
            'video_id': best_video['id'],
            'url': best_video['webVideoUrl'],
            'views': best_video['playCount'],
            'engagement': best_engagement,
            'duration': best_video['videoMeta']['duration']
        }

# Generate QR codes
import qrcode

for qr_data in qr_data_list:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data['url'])
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    output_path = os.path.join(qr_output_dir, qr_data['filename'])
    img.save(output_path)

    print(f"✓ Generated QR code: {qr_data['filename']}")

print(f"✓ Generated {len(qr_data_list)} QR codes")
```

### **Step 2: Add QR Code Fields to Excel Output**

**Location**: After PAGE 3 creative reports section, before Excel write

Insert at end of Excel data building (around line 550):

```python
# PAGE 4: VISUAL EXAMPLES (QR CODES)
tab_data.append(['', ''])
tab_data.append(['PAGE_4_VISUAL_EXAMPLES', ''])
tab_data.append(['', ''])

# Output QR code metadata for each winning bucket
for i, bucket in enumerate(winning_buckets, 1):
    if bucket in qr_metadata:
        qr = qr_metadata[bucket]
        tab_data.append([f'QR_BUCKET_{i}_NAME', bucket])
        tab_data.append([f'QR_BUCKET_{i}_FILE', f"{args.hashtag}_{bucket}_top.png"])
        tab_data.append([f'QR_BUCKET_{i}_URL', qr['url']])
        tab_data.append([f'QR_BUCKET_{i}_VIEWS', format_views(qr['views'])])
        tab_data.append([f'QR_BUCKET_{i}_ENGAGEMENT', str(qr['engagement'])])
        tab_data.append([f'QR_BUCKET_{i}_DURATION', f"{qr['duration']}s"])
        tab_data.append(['', ''])
```

### **Step 3: Update Console Output**

**Location**: Final print statement (around line 565)

Update success message:

```python
print(f"\n✅ Extraction complete!")
print(f"  📁 Excel: {excel_path}")
print(f"  📁 QR codes: {qr_output_dir} ({len(qr_data_list)} files)")  # NEW LINE
print(f"  📊 Total fields: {len(tab_data)}")
print(f"  🎨 Creative formulas: {len(formula_names)}")
```

---

## New Excel Fields

**PAGE 4: VISUAL EXAMPLES** (21 fields total):
- `PAGE_4_VISUAL_EXAMPLES` (divider)
- Per bucket (3 buckets × 6 fields = 18):
  - `QR_BUCKET_1_NAME` → "60-90s"
  - `QR_BUCKET_1_FILE` → "wellnesspt2_test5_60-90s_top.png"
  - `QR_BUCKET_1_URL` → "https://www.tiktok.com/@user/video/123"
  - `QR_BUCKET_1_VIEWS` → "820K"
  - `QR_BUCKET_1_ENGAGEMENT` → "1.5"
  - `QR_BUCKET_1_DURATION` → "45s"
  - (empty row)
  - `QR_BUCKET_2_NAME` → "18-33s"
  - `QR_BUCKET_2_FILE` → "wellnesspt2_test5_18-33s_top.png"
  - ...
  - `QR_BUCKET_3_NAME` → "33-60s"
  - `QR_BUCKET_3_FILE` → "wellnesspt2_test5_33-60s_top.png"
  - ...

**Total field increase**: 129 → 150 fields (+21)

---

## Expected Output

### Console:
```bash
$ python extract_client_data.py --client rollo_test5 --hashtag wellnesspt2_test5

📊 Extracting Client Executive Report for #wellnesspt2_test5
✓ Winning buckets: 60-90s, 18-33s, 33-60s
✓ Coverage: 53.3%
✓ Total videos scraped: 10620

📱 Generating QR codes...
✓ Generated QR code: wellnesspt2_test5_60-90s_top.png
✓ Generated QR code: wellnesspt2_test5_18-33s_top.png
✓ Generated QR code: wellnesspt2_test5_33-60s_top.png
✓ Generated 3 QR codes

✅ Extraction complete!
  📁 Excel: .../wellnesspt2_test5_client_data.xlsx
  📁 QR codes: .../qr_codes (3 files)
  📊 Total fields: 150
```

### Files:
```
/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/
├── wellnesspt2_test5_client_data.xlsx (150 fields)
└── qr_codes/
    ├── wellnesspt2_test5_60-90s_top.png (~800 bytes)
    ├── wellnesspt2_test5_18-33s_top.png (~800 bytes)
    └── wellnesspt2_test5_33-60s_top.png (~800 bytes)
```

---

## Dependencies

**Already installed**:
- `qrcode[pil]` - QR code generation (installed in previous Report 2 implementation)
- `calculate_engagement_metrics()` - Already exists in file

**No new dependencies needed**.

---

## Testing

**Test command**:
```bash
cd /home/jorge/rumiaifinal
source venv/bin/activate
python extract_client_data.py --client rollo_test5 --hashtag wellnesspt2_test5 --mode top --strategy contrastive
```

**Validation**:
1. ✅ 3 QR code PNG files created in `qr_codes/`
2. ✅ QR codes are ~700-900 bytes each
3. ✅ Excel has PAGE_4 with 21 new fields
4. ✅ QR_BUCKET_1/2/3 fields populated with actual data
5. ✅ Console shows "QR codes: .../qr_codes (3 files)"

---

## Notes

### Why 3 QR Codes?
- Client executives need to see **one example per winning bucket**
- Allows quick visual assessment of what works in each duration category
- Consistent with business need to show variety across duration buckets

### Selection Logic
- **Highest engagement** ensures best examples
- Engagement = (likes + comments + shares + saves) / views × 100
- Uses same logic as ranking in "TIER1_BUCKET" fields

### Why Not Bottom Performers?
- Report 1 is executive-facing (strategic overview)
- Executives need **aspirational examples**, not cautionary ones
- Report 2 (Creator) includes bottom performers for creator education
