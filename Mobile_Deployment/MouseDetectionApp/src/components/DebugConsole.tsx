import React, { useEffect, useRef } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

export interface DebugLog {
  time: string;
  message: string;
  type: 'info' | 'warn' | 'error' | 'detection';
}

interface Props {
  logs: DebugLog[];
  maxHeight?: number;
  /** When true the console uses flex:1 to fill remaining space (landscape sidebar) */
  flex?: boolean;
}

const LOG_COLORS: Record<DebugLog['type'], string> = {
  info: '#8be9fd',
  warn: '#f1fa8c',
  error: '#ff5555',
  detection: '#50fa7b',
};

export const DebugConsole: React.FC<Props> = ({ logs, maxHeight = 150, flex = false }) => {
  const scrollRef = useRef<ScrollView>(null);

  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({ animated: false }), 50);
  }, [logs.length]);

  return (
    <View style={[styles.container, flex ? { flex: 1 } : { maxHeight }]}>
      <View style={styles.titleBar}>
        <Text style={styles.titleText}>📋 Debug Console</Text>
        <Text style={styles.countText}>{logs.length} logs</Text>
      </View>
      <ScrollView
        ref={scrollRef}
        style={styles.scrollArea}
        showsVerticalScrollIndicator={true}>
        {logs.map((log, i) => (
          <Text
            key={i}
            style={[styles.logLine, { color: LOG_COLORS[log.type] }]}>
            <Text style={styles.logTime}>[{log.time}] </Text>
            {log.message}
          </Text>
        ))}
        {logs.length === 0 && (
          <Text style={styles.placeholder}>等待日志输出...</Text>
        )}
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'rgba(15, 15, 25, 0.96)',
    borderTopWidth: 1,
    borderTopColor: '#1e1e2e',
  },
  titleBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    backgroundColor: 'rgba(30, 30, 40, 0.95)',
  },
  titleText: {
    color: '#bd93f9',
    fontSize: 11,
    fontWeight: '700',
  },
  countText: {
    color: '#6272a4',
    fontSize: 10,
  },
  scrollArea: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  logLine: {
    fontSize: 10,
    fontFamily: 'Menlo',
    lineHeight: 15,
  },
  logTime: {
    color: '#6272a4',
  },
  placeholder: {
    color: '#6272a4',
    fontSize: 10,
    fontStyle: 'italic',
  },
});
