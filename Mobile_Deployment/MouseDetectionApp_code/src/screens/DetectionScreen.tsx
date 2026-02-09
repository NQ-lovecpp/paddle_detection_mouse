import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';
import { DetectionResult } from '../components/DetectionResult';
import { ImageSelector } from '../components/ImageSelector';
import { ImageProcessor } from '../services/ImageProcessor';
import { ModelService } from '../services/ModelService';
import { Detection } from '../types';

export const DetectionScreen: React.FC = () => {
  const modelService = useMemo(() => new ModelService(), []);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Initializing model...');
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);

  useEffect(() => {
    let mounted = true;
    modelService
      .initialize()
      .then(() => mounted && setStatus('Model ready'))
      .catch(error => mounted && setStatus(`Model error: ${String(error)}`));
    return () => {
      mounted = false;
    };
  }, [modelService]);

  const handleImagePicked = useCallback(
    async (base64: string, uri: string) => {
      setLoading(true);
      setStatus('Running inference...');
      setImageUri(uri);
      setDetections([]);

      try {
        const preprocess = ImageProcessor.preprocessFromBase64(
          base64,
          modelService.getConfig()
        );
        const results = await modelService.detect(preprocess);
        setDetections(results);
        setStatus(`Done. ${results.length} detections`);
      } catch (error) {
        setStatus(`Inference failed: ${String(error)}`);
      } finally {
        setLoading(false);
      }
    },
    [modelService]
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Mouse Detection</Text>
      <Text style={styles.status}>{status}</Text>
      <ImageSelector onImagePicked={handleImagePicked} />

      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.preview} resizeMode="contain" />
      ) : null}

      {loading ? <ActivityIndicator size="large" color="#3b82f6" /> : null}
      <DetectionResult detections={detections} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 20,
  },
  title: {
    fontSize: 26,
    fontWeight: '700',
    color: '#222',
    textAlign: 'center',
  },
  status: {
    marginTop: 6,
    textAlign: 'center',
    color: '#555',
  },
  preview: {
    marginTop: 12,
    width: '100%',
    height: 240,
    backgroundColor: '#eee',
    borderRadius: 8,
  },
});
