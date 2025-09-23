#!/bin/bash
# Setup script to create a dedicated RAM disk for FEAT
# Run this once at system startup or before processing

# Check if running with sufficient permissions
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo ./setup_ramdisk.sh"
    exit 1
fi

# Configuration
RAMDISK_SIZE="512M"  # Adjust based on your needs
MOUNT_POINT="/mnt/feat_ramdisk"

echo "Setting up RAM disk for FEAT..."

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    mkdir -p "$MOUNT_POINT"
    echo "✓ Created mount point: $MOUNT_POINT"
fi

# Check if already mounted
if mount | grep -q "$MOUNT_POINT"; then
    echo "⚠ RAM disk already mounted at $MOUNT_POINT"
    echo "  To remount, first run: sudo umount $MOUNT_POINT"
else
    # Mount RAM disk
    mount -t tmpfs -o size=$RAMDISK_SIZE tmpfs "$MOUNT_POINT"

    # Set permissions so your user can write
    chmod 777 "$MOUNT_POINT"

    echo "✓ Mounted ${RAMDISK_SIZE} RAM disk at $MOUNT_POINT"
fi

# Create FEAT temp directory
FEAT_DIR="$MOUNT_POINT/feat_temp"
mkdir -p "$FEAT_DIR"
chmod 777 "$FEAT_DIR"

echo "✓ FEAT temp directory ready at: $FEAT_DIR"
echo ""
echo "RAM Disk Statistics:"
df -h "$MOUNT_POINT"
echo ""
echo "To use in Python, set:"
echo "  temp_dir = '$FEAT_DIR'"
echo ""
echo "To unmount when done:"
echo "  sudo umount $MOUNT_POINT"