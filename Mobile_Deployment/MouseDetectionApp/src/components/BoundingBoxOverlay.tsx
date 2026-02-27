import React, { memo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Detection } from '../types';

interface Props {
  detections: Detection[];
  // The camera preview dimensions
  previewWidth: number;
  previewHeight: number;
  // The original image dimensions that detections are relative to
  imageWidth: number;
  imageHeight: number;
}

// Memoized: only re-renders when detections or layout actually changes
export const BoundingBoxOverlay: React.FC<Props> = memo(({
  detections,
  previewWidth,
  previewHeight,
  imageWidth,
  imageHeight,
}) => {
  if (!detections.length || !previewWidth || !previewHeight) {
    return null;
  }

  // Scale factors from original image coords to preview coords
  const scaleX = previewWidth / imageWidth;
  const scaleY = previewHeight / imageHeight;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {detections.map((det, index) => {
        const left = det.bbox.x1 * scaleX;
        const top = det.bbox.y1 * scaleY;
        const width = (det.bbox.x2 - det.bbox.x1) * scaleX;
        const height = (det.bbox.y2 - det.bbox.y1) * scaleY;

        return (
          <View
            key={`${det.classId}-${index}`}
            style={[
              styles.box,
              {
                left,
                top,
                width,
                height,
              },
            ]}>
            <View style={styles.label}>
              <Text style={styles.labelText}>
                {det.className} {(det.confidence * 100).toFixed(0)}%
              </Text>
            </View>
          </View>
        );
      })}
    </View>
  );
});

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: '#00FF00',
    borderRadius: 2,
  },
  label: {
    position: 'absolute',
    top: -20,
    left: -1,
    backgroundColor: '#00FF00',
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 2,
  },
  labelText: {
    color: '#000',
    fontSize: 11,
    fontWeight: '700',
  },
});
