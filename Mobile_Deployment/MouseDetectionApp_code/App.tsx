import React from 'react';
import { SafeAreaView, StatusBar, StyleSheet } from 'react-native';
import './src/utils/setup';
import { DetectionScreen } from './src/screens/DetectionScreen';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <DetectionScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f7f7f7',
  },
});
