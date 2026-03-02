import React from 'react';
import { StyleSheet, View } from 'react-native';
import Slider from '@react-native-community/slider';

interface Props {
  value: number;
  onValueChange: (val: number) => void;
}

export const ThresholdSlider: React.FC<Props> = ({ value, onValueChange }) => (
  <View style={styles.container}>
    <Slider
      style={styles.slider}
      minimumValue={0.05}
      maximumValue={0.95}
      step={0.05}
      value={value}
      onValueChange={onValueChange}
      minimumTrackTintColor="#22c55e"
      maximumTrackTintColor="#334155"
      thumbTintColor="#22c55e"
    />
  </View>
);

const styles = StyleSheet.create({
  container: {
    marginHorizontal: -4, // offset slider's internal padding so it aligns with sidebar edges
  },
  slider: {
    width: '100%',
    height: 28,
  },
});
