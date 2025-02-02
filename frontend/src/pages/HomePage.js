import React, { useState, useEffect, useRef } from 'react';
import '../styles/HomePage.css';
import videoIcon from '../assets/video-icon.png';
import realtimeIcon from '../assets/realtime-icon.png';
import fileIcon from '../assets/file-icon.png';
import processingIcon from '../assets/processing-icon.png';
import reloadIcon from '../assets/reload-icon.png';
import expandIcon from '../assets/expand-icon.png';

const HomePage = () => {
  const [resultMedia, setResultMedia] = useState(null);
  const [originalMedia, setOriginalMedia] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [showMoreOriginal, setShowMoreOriginal] = useState(false);
  const [showMoreAnnotated, setShowMoreAnnotated] = useState(false);
  const [processingInfo, setProcessingInfo] = useState(null);
  const [intermediateImages, setIntermediateImages] = useState({});
  const [isExpandedOriginal, setIsExpandedOriginal] = useState(false);
  const [isExpandedAnnotated, setIsExpandedAnnotated] = useState(false);
  const [mediaType, setMediaType] = useState(''); // New state to track media type
  const [timer, setTimer] = useState(0);
  const [milliseconds, setMilliseconds] = useState(0);
  const [finalTime, setFinalTime] = useState(null);
  const [processingMethod, setProcessingMethod] = useState('opencv'); // New state for processing method
  const startTimeRef = useRef(null);

  useEffect(() => {
    let interval;
    if (isProcessing) {
      startTimeRef.current = Date.now();
      interval = setInterval(() => {
        const elapsedTime = Date.now() - startTimeRef.current;
        setTimer(Math.floor(elapsedTime / 1000));
        setMilliseconds(elapsedTime % 1000);
      }, 10);
    } else {
      setTimer(0);
      setMilliseconds(0);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  useEffect(() => {
    if (!isProcessing && isUploaded) {
      const elapsedTime = Date.now() - startTimeRef.current;
      setFinalTime(`${Math.floor(elapsedTime / 1000)} seconds ${elapsedTime % 1000} milliseconds`);
    }
  }, [isProcessing, isUploaded]);

  const handleFileClick = (type) => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = type === 'video' ? 'video/*' : 'image/*';
    fileInput.onchange = async (event) => {
      const files = event.target.files;
      if (files.length > 0) {
        const formData = new FormData();
        formData.append('file', files[0]);
        const fileURL = URL.createObjectURL(files[0]);
        console.log('File URL:', fileURL); // Debugging: Check the file URL
        setOriginalMedia(fileURL); // Ensure originalMedia is set
        setMediaType(type); // Set the media type
        setIsProcessing(true);
        setIsUploaded(true);

        try {
          const endpoint = type === 'video' 
            ? (processingMethod === 'tensorflow' ? 'upload_video_tensorflow' : 'upload_video') 
            : (processingMethod === 'tensorflow' ? 'upload_image_tensorflow' : 'upload');
          const response = await fetch(`http://172.20.10.10:8000/api/${endpoint}`, {
            method: 'POST',
            body: formData,
          });
          const result = await response.json();
          console.log('Upload result:', result);
          if (result.status === 'success') {
            setResultMedia(`http://172.20.10.10:8000${encodeURI(result.result_url)}`);
            setProcessingInfo(result); // Store processing info
            setIntermediateImages(result.intermediate_images); // Store intermediate images
          }
        } catch (error) {
          console.error('Error uploading file:', error);
        } finally {
          setIsProcessing(false);
        }
      }
    };
    fileInput.click();
  };

  const handleOptionChange = (event) => {
    setSelectedOption(event.target.value);
    setProcessingMethod(event.target.value); // Set processing method based on dropdown selection
  };

  const handleReloadClick = async () => {
    if (originalMedia) {
      setIsProcessing(true);
      setIsUploaded(false); // Hide the result page
      try {
        const formData = new FormData();
        const response = await fetch(originalMedia);
        const blob = await response.blob();
        formData.append('file', blob, 'media'); // Use the same media file
        const endpoint = blob.type.startsWith('video') 
          ? (processingMethod === 'tensorflow' ? 'upload_video_tensorflow' : 'upload_video') 
          : (processingMethod === 'tensorflow' ? 'upload_image_tensorflow' : 'upload');
        const uploadResponse = await fetch(`http://172.20.10.10:8000/api/${endpoint}`, {
          method: 'POST',
          body: formData,
        });
        const result = await uploadResponse.json();
        console.log('Reload result:', result);
        if (result.status === 'success') {
          setResultMedia(`http://172.20.10.10:8000${encodeURI(result.result_url)}`);
          setProcessingInfo(result); // Store processing info
          setIntermediateImages(result.intermediate_images); // Store intermediate images
        }
      } catch (error) {
        console.error('Error reloading file:', error);
      } finally {
        setIsProcessing(false);
        setIsUploaded(true); // Show the result page again
      }
    }
  };

  const handleMoreClick = (type) => {
    if (type === 'original') {
      setShowMoreOriginal(!showMoreOriginal);
    } else if (type === 'annotated') {
      setShowMoreAnnotated(!showMoreAnnotated);
    }
  };

  const handleCloseClick = (type) => {
    if (type === 'original') {
      setShowMoreOriginal(false);
    } else if (type === 'annotated') {
      setShowMoreAnnotated(false);
    }
  };

  const handleExpandClick = (type) => {
    if (type === 'original') {
      setIsExpandedOriginal(!isExpandedOriginal);
      document.querySelector('.result-page').classList.toggle('blurred-border', !isExpandedOriginal);
    } else if (type === 'annotated') {
      setIsExpandedAnnotated(!isExpandedAnnotated);
      document.querySelector('.result-page').classList.toggle('blurred-border', !isExpandedAnnotated);
    }
  };

  const getMediaHeight = (mediaId) => {
    const media = document.getElementById(mediaId);
    return media ? media.clientHeight : 'auto';
  };

  const getMediaWidth = (mediaId) => {
    const media = document.getElementById(mediaId);
    return media ? media.clientWidth : 'auto';
  };

  const handlePlayClick = () => {
    const originalVideo = document.getElementById('originalMedia');
    const annotatedVideo = document.getElementById('annotatedMedia');
    console.log('Original Media URL:', originalMedia); // Debugging: Check the original media URL
    console.log('Annotated Media URL:', resultMedia); // Debugging: Check the annotated media URL
    if (originalVideo && annotatedVideo) {
      originalVideo.currentTime = 0;
      annotatedVideo.currentTime = 0;
      originalVideo.play().catch(error => console.error('Error playing original video:', error));
      annotatedVideo.play().catch(error => console.error('Error playing annotated video:', error));
    }
  };

  useEffect(() => {
    console.log('Original Media URL:', originalMedia); // Debugging: Check the original media URL
  }, [originalMedia]);

  if (isProcessing) {
    return (
      <div className="processing-page">
        <h2>Processing...</h2>
        <div className="processing-icon-container">
          <img src={processingIcon} alt="Processing Icon" className="processing-icon" />
        </div>
        <p>Time elapsed: {timer} seconds {milliseconds} milliseconds</p> {/* Add timer display */}
      </div>
    );
  }

  if (isUploaded && resultMedia) {
    return (
      <div className="result-page" style={{ height: showMoreOriginal || showMoreAnnotated ? 'auto' : '100vh' }}>
        <div className="four-img-container"> {/* New container */}
          <div className="result-images">
            <div className="image-container">
              <h3>Original {mediaType === 'video' ? 'Video' : 'Image'}:</h3>
              {originalMedia ? (
                mediaType === 'video' ? (
                  <video id="originalMedia" src={originalMedia} controls />
                ) : (
                  <img id="originalMedia" src={originalMedia} alt="Original" />
                )
              ) : (
                <div className="blank-canvas"></div> // Show blank canvas if original media is not available
              )}
              <button className="more-btn" onClick={() => handleMoreClick('original')}>
                More <span className="arrow">{showMoreOriginal ? '▲' : '▼'}</span>
              </button>
              {showMoreOriginal && (
                <div className={`more-window ${isExpandedOriginal ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedOriginal ? getMediaHeight('originalMedia') * 1.3 : getMediaHeight('originalMedia'), width: isExpandedOriginal ? getMediaWidth('originalMedia') * 1.3 : getMediaWidth('originalMedia') }}>
                  <button className="close-btn" onClick={() => handleCloseClick('original')}>×</button>
                  <button className="expand-btn" onClick={() => handleExpandClick('original')}>
                    <img src={expandIcon} alt="Expand Icon" style={{ width: '20px', height: '20px', filter: 'invert(100%) sepia(100%) saturate(0%) hue-rotate(60deg) brightness(100%) contrast(100%)' }} />
                  </button>
                  {intermediateImages && (
                    <div className="intermediate-images">
                      <p><strong>Gray Scale:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.gray}`} alt="Gray Scale" />
                      <p><strong>Edge Detection:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.edge}`} alt="Edge Detection" />
                      <p><strong>Localized Image:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.localized}`} alt="Localized" />
                      <p><strong>License Plate:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.plate}`} alt="License Plate" />
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="image-container">
              <h3>Annotated {mediaType === 'video' ? 'Video' : 'Image'}:</h3>
              {resultMedia ? (
                mediaType === 'video' ? (
                  <video id="annotatedMedia" src={resultMedia} controls />
                ) : (
                  <img id="annotatedMedia" src={resultMedia} alt="Annotated result" />
                )
              ) : (
                <div className="blank-canvas"></div> // Show blank canvas if result media is not available
              )}
              <button className="more-btn" onClick={() => handleMoreClick('annotated')}>
                More <span className="arrow">{showMoreAnnotated ? '▲' : '▼'}</span>
              </button>
              {showMoreAnnotated && (
                <div className={`more-window ${isExpandedAnnotated ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedAnnotated ? getMediaHeight('annotatedMedia') * 1.3 : getMediaHeight('annotatedMedia'), width: isExpandedAnnotated ? getMediaWidth('annotatedMedia') * 1.3 : getMediaWidth('annotatedMedia') }}>
                  <button className="close-btn" onClick={() => handleCloseClick('annotated')}>×</button>
                  <button className="expand-btn" onClick={() => handleExpandClick('annotated')}>
                    <img src={expandIcon} alt="Expand Icon" style={{ width: '20px', height: '20px', filter: 'invert(100%) sepia(100%) saturate(0%) hue-rotate(60deg) brightness(100%) contrast(100%)' }} />
                  </button>
                  {processingInfo && (
                    <div className="processing-info">
                      <p><strong>Filename:</strong> {processingInfo.filename}</p>
                      <p><strong>License Plate:</strong> {processingInfo.license_plate}</p>
                      <p><strong>Status:</strong> {processingInfo.status}</p>
                      <p><strong>Result URL:</strong> <a href={processingInfo.result_url} target="_blank" rel="noopener noreferrer">{processingInfo.result_url}</a></p>
                      <p><strong>Visit History:</strong> this could be the cars visit history to the company</p>
                      {processingInfo.customer_data && (
                        <div className="customer-data">
                          <h4>Customer Data:</h4>
                          <table>
                            <thead>
                              <tr>
                                <th>CustomerID</th>
                                <th>FirstName</th>
                                <th>LastName</th>
                                <th>Email</th>
                                <th>Phone</th>
                                <th>City</th>
                                <th>Country</th>
                              </tr>
                            </thead>
                            <tbody>
                              {processingInfo.customer_data.map((customer, index) => (
                                <tr key={index}>
                                  <td>{customer.CustomerID}</td>
                                  <td>{customer.FirstName}</td>
                                  <td>{customer.LastName}</td>
                                  <td>{customer.Email}</td>
                                  <td>{customer.Phone}</td>
                                  <td>{customer.City}</td>
                                  <td>{customer.Country}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          {mediaType === 'video' && (
            <button className="play-btn" onClick={handlePlayClick}>Play Both Videos</button>
          )} {/* Conditionally render play button */}
          <p>Processing time: {finalTime}</p> {/* Display final processing time */}
        </div>
        <div className="space-between"></div> {/* Add space */}
        <div className="dropdown-container">
          <div className="icon-container" onClick={handleReloadClick}>
            <img src={reloadIcon} alt="Option Icon" className="dropdown-icon" />
          </div>
          <select id="options" value={selectedOption} onChange={handleOptionChange}>
            <option value=""></option>
            <option value="opencv">OpenCV</option> {/* Add OpenCV option */}
            <option value="tensorflow">TensorFlow</option> {/* Add TensorFlow option */}
          </select>
        </div>
      </div>
    );
  }

  return (
    <div className="home-page">
      <div className="upload-section-container">
        <div className="upload-header">
          <h2>Upload Footage.</h2>
        </div>
        <div className="upload-buttons">
          <button className="upload-btn file" onClick={() => handleFileClick('image')}>
            <img src={fileIcon} alt="File Icon" className="file-icon" />
            <span className="tooltip">Upload Image</span>
          </button>
          <button className="upload-btn video" onClick={() => handleFileClick('video')}>
            <img src={videoIcon} alt="Video Icon" className="video-icon" />
            <span className="tooltip">Upload Video</span>
          </button>
          <button className="upload-btn real-time">
            <img src={realtimeIcon} alt="Real Time Icon" className="real-time-icon" />
            <span className="tooltip">Real-Time Detection</span>
          </button>
        </div>
        <div className="dropdown-container"> {/* Add dropdown container */}
          <div className="icon-container" onClick={handleReloadClick}>
            <img src={reloadIcon} alt="Option Icon" className="dropdown-icon" />
          </div>
          <select id="options" value={selectedOption} onChange={handleOptionChange}>
            <option value=""></option>
            <option value="opencv">OpenCV</option> {/* Add OpenCV option */}
            <option value="tensorflow">TensorFlow</option> {/* Add TensorFlow option */}
          </select>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
