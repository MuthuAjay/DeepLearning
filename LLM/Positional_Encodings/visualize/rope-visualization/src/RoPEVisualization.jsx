import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from 'lucide-react';

// Simplified component versions to replace shadcn/ui
const Card = ({ className, children, ...props }) => (
  <div className={`border border-gray-300 rounded-lg ${className || ''}`} {...props}>{children}</div>
);

const CardHeader = ({ className, children, ...props }) => (
  <div className={`p-4 ${className || ''}`} {...props}>{children}</div>
);

const CardTitle = ({ className, children, ...props }) => (
  <h3 className={`text-xl font-bold ${className || ''}`} {...props}>{children}</h3>
);

const CardContent = ({ className, children, ...props }) => (
  <div className={`p-4 pt-0 ${className || ''}`} {...props}>{children}</div>
);

const RoPEVisualization = () => {
  // State
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [showDotProduct, setShowDotProduct] = useState(false);
  const [selectedTokens, setSelectedTokens] = useState([0, 2]); // Default to comparing "the" and "brown"

  // Constants
  const tokens = ["the", "quick", "brown", "fox"];
  const colors = ["#ff6b6b", "#4ecdc4", "#ffd166", "#6a0572"];
  const baseFreq = 10000;
  const animationDuration = 5000; // ms
  
  // Simplified 2D embedding vectors (in a real model these would be high-dimensional)
  // These represent the semantic meaning of each token
  const initialEmbeddings = [
    [0.8, 0.6],   // "the"
    [0.5, 0.85],  // "quick"
    [-0.7, 0.7],  // "brown"
    [0.9, -0.4]   // "fox"
  ];

  useEffect(() => {
    let animationFrame;
    let startTime = null;
    
    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      
      if (elapsed < animationDuration && isPlaying) {
        const progress = Math.min(elapsed / animationDuration, 1);
        setAnimationProgress(progress);
        animationFrame = requestAnimationFrame(animate);
      } else if (isPlaying) {
        setIsPlaying(false);
        setAnimationProgress(1);
      }
    };
    
    if (isPlaying) {
      animationFrame = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [isPlaying]);

  // Calculate rotation angle based on position
  const getRotationAngle = (position, dim) => {
    // In RoPE, each dimension pair has its own frequency
    const theta = position / Math.pow(baseFreq, dim / 2);
    return theta * 360; // Convert to degrees for visualization
  };

  // Calculate rotated embedding based on position and progress
  const getRotatedEmbedding = (embedding, position, progress) => {
    const theta = (getRotationAngle(position, 0) * Math.PI / 180) * progress;
    
    // Rotation matrix application
    const x = embedding[0];
    const y = embedding[1];
    
    const rotatedX = x * Math.cos(theta) - y * Math.sin(theta);
    const rotatedY = x * Math.sin(theta) + y * Math.cos(theta);
    
    return [rotatedX, rotatedY];
  };

  // Calculate dot product between two vectors
  const dotProduct = (vec1, vec2) => {
    return vec1[0] * vec2[0] + vec1[1] * vec2[1];
  };

  // Calculate vector magnitude
  const magnitude = (vec) => {
    return Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
  };

  // Calculate cosine similarity
  const cosineSimilarity = (vec1, vec2) => {
    return dotProduct(vec1, vec2) / (magnitude(vec1) * magnitude(vec2));
  };

  // Scale coordinates for display
  const scale = 100;
  const center = { x: 200, y: 150 };

  // Handle animation controls
  const togglePlay = () => setIsPlaying(!isPlaying);
  const reset = () => {
    setIsPlaying(false);
    setAnimationProgress(0);
  };

  // Handle step navigation
  const steps = [
    "Initial Embeddings",
    "Applying Rotations",
    "Relative Positioning"
  ];

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      if (currentStep === 0) setAnimationProgress(0);
      if (currentStep === 1) {
        setAnimationProgress(1);
        setShowDotProduct(true);
      }
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      if (currentStep === 2) setShowDotProduct(false);
      if (currentStep === 1) setAnimationProgress(0);
    }
  };

  // Get explanation text based on current step
  const getExplanation = () => {
    switch(currentStep) {
      case 0:
        return "Each token starts with an embedding vector that represents its semantic meaning. These are the initial vectors before any positional encoding.";
      case 1:
        return "RoPE applies a rotation to each token's embedding vector. The rotation angle depends on the token's position in the sequence. Later tokens receive larger rotations.";
      case 2:
        return "The key benefit: when comparing rotated embeddings, their dot product automatically encodes relative position. This helps the model understand token relationships.";
      default:
        return "";
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%', backgroundColor: '#f8fafc', padding: '16px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.12)' }}>
      <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px', color: '#1e293b' }}>Rotary Position Embedding (RoPE)</h1>
      
      {/* Step indicator */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <button 
          onClick={prevStep} 
          disabled={currentStep === 0}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            padding: '4px 8px', 
            backgroundColor: '#e2e8f0', 
            borderRadius: '4px',
            opacity: currentStep === 0 ? 0.5 : 1,
            border: 'none',
            cursor: currentStep === 0 ? 'default' : 'pointer'
          }}
        >
          <ChevronLeft size={16} style={{ marginRight: '4px' }} /> Previous
        </button>
        
        <div style={{ fontSize: '18px', fontWeight: '600', color: '#334155' }}>
          {steps[currentStep]}
        </div>
        
        <button 
          onClick={nextStep} 
          disabled={currentStep === steps.length - 1}
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            padding: '4px 8px', 
            backgroundColor: '#e2e8f0', 
            borderRadius: '4px',
            opacity: currentStep === steps.length - 1 ? 0.5 : 1,
            border: 'none',
            cursor: currentStep === steps.length - 1 ? 'default' : 'pointer'
          }}
        >
          Next <ChevronRight size={16} style={{ marginLeft: '4px' }} />
        </button>
      </div>
      
      {/* Explanation */}
      <div style={{ backgroundColor: 'white', padding: '12px', borderRadius: '4px', marginBottom: '16px', color: '#334155' }}>
        {getExplanation()}
      </div>

      {/* Visualization canvas */}
      <div style={{ position: 'relative', flexGrow: 1, border: '1px solid #cbd5e1', borderRadius: '4px', backgroundColor: 'white', overflow: 'hidden' }}>
        {/* Coordinate system */}
        <svg width="100%" height="100%" viewBox="0 0 400 300">
          {/* Axes */}
          <line x1={center.x} y1="10" x2={center.x} y2="290" stroke="#ccc" strokeWidth="1" />
          <line x1="10" y1={center.y} x2="390" y2={center.y} stroke="#ccc" strokeWidth="1" />
          
          {/* Origin point */}
          <circle cx={center.x} cy={center.y} r="3" fill="#333" />
          
          {/* Token embeddings and their rotations */}
          {tokens.map((token, idx) => {
            const initialEmbed = initialEmbeddings[idx];
            const rotatedEmbed = getRotatedEmbedding(initialEmbed, idx, animationProgress);
            
            // Only show rotated embeddings if we're in the appropriate step
            const showRotated = currentStep >= 1;
            
            return (
              <g key={idx}>
                {/* Initial embedding vector */}
                <line 
                  x1={center.x} 
                  y1={center.y} 
                  x2={center.x + initialEmbed[0] * scale} 
                  y2={center.y - initialEmbed[1] * scale} 
                  stroke={colors[idx]} 
                  strokeWidth="2"
                  strokeOpacity={showRotated ? 0.3 : 1}
                  strokeDasharray={showRotated ? "5,5" : "none"}
                />
                
                {/* Rotated embedding vector (if in appropriate step) */}
                {showRotated && (
                  <line 
                    x1={center.x} 
                    y1={center.y} 
                    x2={center.x + rotatedEmbed[0] * scale} 
                    y2={center.y - rotatedEmbed[1] * scale} 
                    stroke={colors[idx]} 
                    strokeWidth="2"
                  />
                )}
                
                {/* Position label */}
                <text 
                  x={center.x + (showRotated ? rotatedEmbed[0] : initialEmbed[0]) * scale + 10} 
                  y={center.y - (showRotated ? rotatedEmbed[1] : initialEmbed[1]) * scale} 
                  fill={colors[idx]}
                  fontWeight="bold"
                >
                  {token} (pos {idx})
                </text>
                
                {/* Rotation arc (visible during rotation) */}
                {currentStep === 1 && (
                  <path 
                    d={`M ${center.x + 20} ${center.y} A 20 20 0 0 1 ${
                      center.x + 20 * Math.cos(getRotationAngle(idx, 0) * Math.PI / 180 * animationProgress)
                    } ${
                      center.y - 20 * Math.sin(getRotationAngle(idx, 0) * Math.PI / 180 * animationProgress)
                    }`} 
                    fill="none" 
                    stroke={colors[idx]} 
                    strokeWidth="1"
                    strokeDasharray="2,2"
                  />
                )}
              </g>
            );
          })}
          
          {/* Dot product visualization (for step 3) */}
          {showDotProduct && (
            <>
              <line 
                x1={center.x} 
                y1={center.y} 
                x2={center.x + getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1)[0] * scale} 
                y2={center.y - getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1)[1] * scale} 
                stroke={colors[selectedTokens[0]]} 
                strokeWidth="3"
              />
              <line 
                x1={center.x} 
                y1={center.y} 
                x2={center.x + getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1)[0] * scale} 
                y2={center.y - getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1)[1] * scale} 
                stroke={colors[selectedTokens[1]]} 
                strokeWidth="3"
              />
              
              {/* Arc showing angle between vectors */}
              <path 
                d={`M ${
                  center.x + 30 * getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1)[0] / magnitude(getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1))
                } ${
                  center.y - 30 * getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1)[1] / magnitude(getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1))
                } A 30 30 0 0 1 ${
                  center.x + 30 * getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1)[0] / magnitude(getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1))
                } ${
                  center.y - 30 * getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1)[1] / magnitude(getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1))
                }`} 
                fill="none" 
                stroke="#333" 
                strokeWidth="1"
                strokeDasharray="3,3"
              />
            </>
          )}
        </svg>
        
        {/* Info overlay for dot product */}
        {showDotProduct && (
          <div style={{ position: 'absolute', bottom: '16px', right: '16px', backgroundColor: 'white', padding: '8px', borderRadius: '4px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <p style={{ fontSize: '14px', marginBottom: '4px' }}>
              Comparing <span style={{color: colors[selectedTokens[0]]}}>{tokens[selectedTokens[0]]}</span> and <span style={{color: colors[selectedTokens[1]]}}>{tokens[selectedTokens[1]]}</span>:
            </p>
            <p style={{ fontSize: '14px' }}>Position difference: {Math.abs(selectedTokens[0] - selectedTokens[1])}</p>
            <p style={{ fontSize: '14px' }}>Similarity: {cosineSimilarity(
              getRotatedEmbedding(initialEmbeddings[selectedTokens[0]], selectedTokens[0], 1),
              getRotatedEmbedding(initialEmbeddings[selectedTokens[1]], selectedTokens[1], 1)
            ).toFixed(3)}</p>
          </div>
        )}
      </div>
      
      {/* Controls for steps that use animation */}
      {currentStep === 1 && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px', marginTop: '16px' }}>
          <button 
            onClick={togglePlay} 
            style={{ 
              padding: '8px', 
              backgroundColor: '#3b82f6', 
              color: 'white', 
              borderRadius: '4px',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <button 
            onClick={reset} 
            style={{ 
              padding: '8px', 
              backgroundColor: '#d1d5db', 
              borderRadius: '4px',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            <RotateCcw size={20} />
          </button>
          <div style={{ width: '256px', backgroundColor: '#e5e7eb', borderRadius: '9999px', height: '8px' }}>
            <div 
              style={{ 
                backgroundColor: '#3b82f6', 
                borderRadius: '9999px', 
                height: '8px',
                width: `${animationProgress * 100}%`
              }}
            ></div>
          </div>
        </div>
      )}
      
      {/* Token comparison selector (for step 3) */}
      {currentStep === 2 && (
        <div style={{ marginTop: '16px', padding: '12px', backgroundColor: 'white', borderRadius: '4px' }}>
          <p style={{ marginBottom: '8px', fontWeight: '500' }}>Compare tokens:</p>
          <div style={{ display: 'flex', gap: '16px' }}>
            <select 
              value={selectedTokens[0]}
              onChange={(e) => setSelectedTokens([parseInt(e.target.value), selectedTokens[1]])}
              style={{ padding: '4px', border: '1px solid #cbd5e1', borderRadius: '4px' }}
            >
              {tokens.map((token, idx) => (
                <option key={idx} value={idx}>{token} (pos {idx})</option>
              ))}
            </select>
            <select 
              value={selectedTokens[1]}
              onChange={(e) => setSelectedTokens([selectedTokens[0], parseInt(e.target.value)])}
              style={{ padding: '4px', border: '1px solid #cbd5e1', borderRadius: '4px' }}
            >
              {tokens.map((token, idx) => (
                <option key={idx} value={idx}>{token} (pos {idx})</option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
};

export default RoPEVisualization;