import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Slider from '@react-native-community/slider';

interface Props {
  value: number;
  onValueChange: (val: number) => void;
}

export const ThresholdSlider: React.FC<Props> = ({ value, onValueChange }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>
        置信度阈值: <Text style={styles.value}>{(value * 100).toFixed(0)}%</Text>
      </Text>
      <Slider
        style={styles.slider}
        minimumValue={0.05}
        maximumValue={0.95}
        step={0.05}
        value={value}
        onValueChange={onValueChange}
        minimumTrackTintColor="#22c55e"
        maximumTrackTintColor="#555"
        thumbTintColor="#22c55e"
      />
      <View style={styles.range}>
        <Text style={styles.rangeText}>5%</Text>
        <Text style={styles.rangeText}>95%</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  label: {
    color: '#ccc',
    fontSize: 12,
    marginBottom: 2,
  },
  value: {
    color: '#22c55e',
    fontWeight: '700',
    fontSize: 13,
  },
  slider: {
    width: '100%',
    height: 30,
  },
  range: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: -4,
  },
  rangeText: {
    color: '#666',
    fontSize: 10,
  },
});
