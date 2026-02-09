import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import RNFS from 'react-native-fs';

interface Props {
  onImagePicked: (base64: string, uri: string) => void;
}

export const ImageSelector: React.FC<Props> = ({ onImagePicked }) => {
  const extractBase64 = async (uri?: string, base64?: string) => {
    if (base64) {
      return base64;
    }
    if (!uri) {
      throw new Error('Missing image data');
    }
    const normalized = uri.startsWith('file://') ? uri.replace('file://', '') : uri;
    return RNFS.readFile(normalized, 'base64');
  };

  const handlePick = async (source: 'library' | 'camera') => {
    const launcher = source === 'camera' ? launchCamera : launchImageLibrary;
    const result = await launcher({
      mediaType: 'photo',
      includeBase64: true,
      quality: 1,
    });

    if (result.didCancel || !result.assets?.length) {
      return;
    }

    const asset = result.assets[0];
    const base64 = await extractBase64(asset.uri, asset.base64);
    if (asset.uri) {
      onImagePicked(base64, asset.uri);
    }
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.button} onPress={() => handlePick('library')}>
        <Text style={styles.buttonText}>选择相册图片</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={() => handlePick('camera')}>
        <Text style={styles.buttonText}>拍照检测</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    marginVertical: 12,
    gap: 10,
  },
  button: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
  },
});
