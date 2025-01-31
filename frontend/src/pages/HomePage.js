import React, { useState, useEffect } from 'react';
import '../styles/HomePage.css';
import videoIcon from '../assets/video-icon.png';
import fileIcon from '../assets/file-icon.png';
import driveIcon from '../assets/drive-icon.png';
import processingIcon from '../assets/processing-icon.png';
import reloadIcon from '../assets/reload-icon.png';
import expandIcon from '../assets/expand-icon.png';

const HomePage = () => {
  const [resultImage, setResultImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [showMoreOriginal, setShowMoreOriginal] = useState(false);
  const [showMoreAnnotated, setShowMoreAnnotated] = useState(false);
  const [processingInfo, setProcessingInfo] = useState(null);
  const [intermediateImages, setIntermediateImages] = useState({});
  const [isExpandedOriginal, setIsExpandedOriginal] = useState(false);
  const [isExpandedAnnotated, setIsExpandedAnnotated] = useState(false);

  const handleFileClick = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = async (event) => {
      const files = event.target.files;
      if (files.length > 0) {
        const formData = new FormData();
        formData.append('file', files[0]);
        setOriginalImage(URL.createObjectURL(files[0]));
        setIsProcessing(true);
        setIsUploaded(true);

        try {
          const response = await fetch('http://localhost:8000/api/upload', {
            method: 'POST',
            body: formData,
          });
          const result = await response.json();
          console.log('Upload result:', result);
          if (result.status === 'success') {
            setResultImage(`http://localhost:8000${result.result_url}`);
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
  };

  const handleReloadClick = async () => {
    if (originalImage) {
      setIsProcessing(true);
      setIsUploaded(false); // Hide the result page
      try {
        const formData = new FormData();
        const response = await fetch(originalImage);
        const blob = await response.blob();
        formData.append('file', blob, 'image.jpg'); // Use the same image file
        const uploadResponse = await fetch('http://localhost:8000/api/upload', {
          method: 'POST',
          body: formData,
        });
        const result = await uploadResponse.json();
        console.log('Reload result:', result);
        if (result.status === 'success') {
          setResultImage(`http://localhost:8000${result.result_url}`);
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
      document.querySelector('.result-page').classList.toggle('blurred-background', !isExpandedOriginal);
    } else if (type === 'annotated') {
      setIsExpandedAnnotated(!isExpandedAnnotated);
      document.querySelector('.result-page').classList.toggle('blurred-background', !isExpandedAnnotated);
    }
  };

  const getImageHeight = (imageId) => {
    const img = document.getElementById(imageId);
    return img ? img.clientHeight : 'auto';
  };

  const getImageWidth = (imageId) => {
    const img = document.getElementById(imageId);
    return img ? img.clientWidth : 'auto';
  };

  if (isProcessing) {
    return (
      <div className="processing-page">
        <h2>Processing...</h2>
        <div className="processing-icon-container">
          <img src={processingIcon} alt="Processing Icon" className="processing-icon" />
        </div>
      </div>
    );
  }

  if (isUploaded && resultImage) {
    return (
      <div className="result-page" style={{ height: showMoreOriginal || showMoreAnnotated ? 'auto' : '100vh' }}>
        <div className="four-img-container"> {/* New container */}
          <div className="result-images">
            <div className="image-container">
              <h3>Original Image:</h3>
              <img id="originalImage" src={originalImage} alt="Original" />
              <button className="more-btn" onClick={() => handleMoreClick('original')}>
                More <span className="arrow">{showMoreOriginal ? '▲' : '▼'}</span>
              </button>
              {showMoreOriginal && (
                <div className={`more-window ${isExpandedOriginal ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedOriginal ? getImageHeight('originalImage') * 1.3 : getImageHeight('originalImage'), width: isExpandedOriginal ? getImageWidth('originalImage') * 1.3 : getImageWidth('originalImage') }}>
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
              <h3>Annotated Image:</h3>
              <img id="annotatedImage" src={resultImage} alt="Annotated result" />
              <button className="more-btn" onClick={() => handleMoreClick('annotated')}>
                More <span className="arrow">{showMoreAnnotated ? '▲' : '▼'}</span>
              </button>
              {showMoreAnnotated && (
                <div className={`more-window ${isExpandedAnnotated ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedAnnotated ? getImageHeight('annotatedImage') * 1.3 : getImageHeight('annotatedImage'), width: isExpandedAnnotated ? getImageWidth('annotatedImage') * 1.3 : getImageWidth('annotatedImage') }}>
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
        </div>
        <div className="space-between"></div> {/* Add space */}
        <div className="dropdown-container">
          <div className="icon-container" onClick={handleReloadClick}>
            <img src={reloadIcon} alt="Option Icon" className="dropdown-icon" />
          </div>
          <select id="options" value={selectedOption} onChange={handleOptionChange}>
            <option value=""></option>
            <option value="option1">Option 1</option>
            <option value="option2">Option 2</option>
            <option value="option3">Option 3</option>
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
          <button className="upload-btn url">
            <img src={videoIcon} alt="Url Icon" className="url-icon" />
          </button>
          <button className="upload-btn file" onClick={handleFileClick}>
            <img src={fileIcon} alt="File Icon" className="file-icon" />
          </button>
          <button className="upload-btn drive">
            <img src={driveIcon} alt="Drive Icon" className="drive-icon" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
