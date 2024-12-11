import React, { useState } from 'react';
import { Upload, Button, Progress, Card, Alert, Row, Col, Typography, Descriptions } from 'antd';
import { UploadOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { uploadVideo } from '../services/api';

const { Title, Text } = Typography;

const DeepFakeDetector = () => {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileUpload = async (info) => {
    const file = info.file;
    setFile(file);
    setError(null);
    setResults(null);

    try {
      const response = await uploadVideo(file, (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setUploadProgress(percentCompleted);
      });

      // Simulate processing stages
      setProcessingProgress(25);
      setTimeout(() => {
        setProcessingProgress(50);
        setTimeout(() => {
          setProcessingProgress(75);
          setTimeout(() => {
            setProcessingProgress(100);
            setResults(response.data);
          }, 500);
        }, 500);
      }, 500);

    } catch (error) {
      setError('Video processing failed');
      setUploadProgress(0);
      setProcessingProgress(0);
    }
  };

  return (
    <div style={{ maxWidth: 1000, margin: 'auto', padding: 20 }}>
      <Card>
        <Title level={2} style={{ textAlign: 'center', marginBottom: 20 }}>
          DeepFake Video Detector
        </Title>

        <Upload
          beforeUpload={() => false}
          onChange={handleFileUpload}
          maxCount={1}
          accept=".mp4,.avi,.mov"
        >
          <Button icon={<UploadOutlined />} size="large">
            Select Video
          </Button>
        </Upload>

        {uploadProgress > 0 && (
          <div style={{ marginTop: 16 }}>
            <Text strong>Upload Progress</Text>
            <Progress percent={uploadProgress} />
          </div>
        )}

        {processingProgress > 0 && (
          <div style={{ marginTop: 16 }}>
            <Text strong>Processing Progress</Text>
            <Progress percent={processingProgress} />
          </div>
        )}

        {results && (
          <div style={{ marginTop: 16 }}>
            <Alert
              message={results.is_fake ? 'Detected as FAKE' : 'Detected as REAL'}
              type={results.is_fake ? 'error' : 'success'}
              icon={results.is_fake ? <CloseCircleOutlined /> : <CheckCircleOutlined />}
            />
            <Card
              title="Detection Details"
              style={{ marginTop: 16 }}
            >
              <Descriptions bordered column={2}>
                <Descriptions.Item label="Fake Probability">
                  {(results.fake_probability * 100).toFixed(2)}%
                </Descriptions.Item>
                <Descriptions.Item label="Total Faces Detected">
                  {results.total_faces_detected}
                </Descriptions.Item>
                <Descriptions.Item label="Fake Frame Count">
                  {results.fake_frame_count}
                </Descriptions.Item>
                <Descriptions.Item label="Consistent Fake Frames">
                  {results.consistent_fake_frames}
                </Descriptions.Item>
              </Descriptions>

              <Row gutter={16} style={{ marginTop: 16 }}>
                <Col span={12}>
                  <Title level={4}>Extracted Frames</Title>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                    {results.frame_paths.map((framePath, index) => (
                      <img
                        key={index}
                        src={`http://localhost:5000${framePath}`}
                        alt={`Frame ${index + 1}`}
                        style={{ width: 150, height: 100, objectFit: 'cover' }}
                      />
                    ))}
                  </div>
                </Col>
                <Col span={12}>
                  <Title level={4}>Detected Faces</Title>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                    {results.face_paths.map((facePath, index) => (
                      <img
                        key={index}
                        src={`http://localhost:5000${facePath}`}
                        alt={`Face ${index + 1}`}
                        style={{ width: 150, height: 100, objectFit: 'cover' }}
                      />
                    ))}
                  </div>
                </Col>
              </Row>
            </Card>
          </div>
        )}

        {error && (
          <Alert
            message={error}
            type="error"
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
    </div