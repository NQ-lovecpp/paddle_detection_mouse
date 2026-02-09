import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Detection } from '../types';

interface Props {
  detections: Detection[];
}

export const DetectionResult: React.FC<Props> = ({ detections }) => {
  if (!detections.length) {
    return <Text style={styles.empty}>No detections yet.</Text>;
  }

  return (
    <View style={styles.container}>
      {detections.map((det, index) => (
        <View key={`${det.classId}-${index}`} style={styles.item}>
          <Text style={styles.title}>{det.className}</Text>
          <Text style={styles.meta}>Confidence: {det.confidence.toFixed(3)}</Text>
          <Text style={styles.meta}>
            Box: {det.bbox.x1.toFixed(1)}, {det.bbox.y1.toFixed(1)} - {det.bbox.x2.toFixed(1)},{' '}
            {det.bbox.y2.toFixed(1)}
          </Text>
        </View>
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    marginTop: 16,
  },
  empty: {
    marginTop: 16,
    color: '#888',
    textAlign: 'center',
  },
  item: {
    backgroundColor: '#fff',
    padding: 12,
    marginBottom: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#eee',
  },
  title: {
    fontWeight: '600',
    fontSize: 16,
    color: '#222',
  },
  meta: {
    color: '#555',
    marginTop: 4,
  },
});
