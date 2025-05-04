import base64
import os

def get_encoded_gif(gif_type):
    """
    Get base64 encoded GIF for display in Streamlit
    
    Parameters:
    -----------
    gif_type : str
        Type of GIF to return ('welcome', 'completion', etc.)
        
    Returns:
    --------
    encoded_gif : str
        Base64 encoded GIF
    """
    # SVG-based animations for finance themed GIFs
    if gif_type == 'welcome':
        # Create an SVG animation of a rising stock chart
        svg_code = """
        <svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="500" height="300" fill="#f8f9fa" rx="10" ry="10"/>
            
            <!-- Chart grid lines -->
            <g stroke="#e0e0e0" stroke-width="1">
                <line x1="50" y1="50" x2="50" y2="250" />
                <line x1="50" y1="250" x2="450" y2="250" />
                <line x1="50" y1="90" x2="450" y2="90" />
                <line x1="50" y1="130" x2="450" y2="130" />
                <line x1="50" y1="170" x2="450" y2="170" />
                <line x1="50" y1="210" x2="450" y2="210" />
            </g>
            
            <!-- Y-axis labels -->
            <text x="30" y="250" font-family="Arial" font-size="12" text-anchor="end" fill="#666">0</text>
            <text x="30" y="210" font-family="Arial" font-size="12" text-anchor="end" fill="#666">20</text>
            <text x="30" y="170" font-family="Arial" font-size="12" text-anchor="end" fill="#666">40</text>
            <text x="30" y="130" font-family="Arial" font-size="12" text-anchor="end" fill="#666">60</text>
            <text x="30" y="90" font-family="Arial" font-size="12" text-anchor="end" fill="#666">80</text>
            <text x="30" y="50" font-family="Arial" font-size="12" text-anchor="end" fill="#666">100</text>
            
            <!-- X-axis labels -->
            <text x="50" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Jan</text>
            <text x="130" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Feb</text>
            <text x="210" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Mar</text>
            <text x="290" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Apr</text>
            <text x="370" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">May</text>
            <text x="450" y="270" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">Jun</text>
            
            <!-- Animated stock chart line -->
            <path id="chartLine" d="M50,250 L50,250" fill="none" stroke="#2E86C1" stroke-width="3">
                <animate 
                    attributeName="d" 
                    from="M50,250 L50,250" 
                    to="M50,220 L130,200 L210,180 L290,140 L370,90 L450,70" 
                    dur="2s" 
                    begin="0s"
                    fill="freeze"/>
            </path>
            
            <!-- Animated area under the curve -->
            <path id="chartArea" d="M50,250 L50,250 L50,250 Z" fill="#2E86C1" fill-opacity="0.2" stroke="none">
                <animate 
                    attributeName="d" 
                    from="M50,250 L50,250 L50,250 Z" 
                    to="M50,220 L130,200 L210,180 L290,140 L370,90 L450,70 L450,250 L50,250 Z" 
                    dur="2s" 
                    begin="0s"
                    fill="freeze"/>
            </path>
            
            <!-- Animated points on the line -->
            <circle cx="50" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="220" dur="2s" begin="0s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="0s" fill="freeze"/>
            </circle>
            
            <circle cx="130" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="200" dur="2s" begin="0.3s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="0.3s" fill="freeze"/>
            </circle>
            
            <circle cx="210" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="180" dur="2s" begin="0.6s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="0.6s" fill="freeze"/>
            </circle>
            
            <circle cx="290" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="140" dur="2s" begin="0.9s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="0.9s" fill="freeze"/>
            </circle>
            
            <circle cx="370" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="90" dur="2s" begin="1.2s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="1.2s" fill="freeze"/>
            </circle>
            
            <circle cx="450" cy="250" r="0" fill="#2E86C1">
                <animate attributeName="cy" from="250" to="70" dur="2s" begin="1.5s" fill="freeze"/>
                <animate attributeName="r" from="0" to="5" dur="0.5s" begin="1.5s" fill="freeze"/>
            </circle>
            
            <!-- Title with animation -->
            <text id="title" x="250" y="30" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle" fill="#2E86C1" opacity="0">
                Financial ML Pipeline
                <animate attributeName="opacity" from="0" to="1" dur="1s" begin="1.7s" fill="freeze"/>
            </text>
        </svg>
        """
        
    elif gif_type == 'completion':
        # Create an SVG animation of a completed analysis with a checkmark
        svg_code = """
        <svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="500" height="300" fill="#f8f9fa" rx="10" ry="10"/>
            
            <!-- Analytics Dashboard Background -->
            <rect x="50" y="50" width="400" height="200" fill="#fff" stroke="#e0e0e0" stroke-width="1" rx="5" ry="5"/>
            
            <!-- Chart 1: Bar Chart (Pre-animated) -->
            <g transform="translate(80, 80) scale(0.7)">
                <rect x="0" y="40" width="20" height="60" fill="#3498DB"/>
                <rect x="30" y="30" width="20" height="70" fill="#3498DB"/>
                <rect x="60" y="10" width="20" height="90" fill="#3498DB"/>
                <rect x="90" y="20" width="20" height="80" fill="#3498DB"/>
                <rect x="120" y="0" width="20" height="100" fill="#3498DB"/>
            </g>
            
            <!-- Chart 2: Line Chart (Pre-animated) -->
            <g transform="translate(250, 130) scale(0.7)">
                <polyline points="0,50 30,30 60,40 90,10 120,20" fill="none" stroke="#2ECC71" stroke-width="3"/>
                <circle cx="0" cy="50" r="4" fill="#2ECC71"/>
                <circle cx="30" cy="30" r="4" fill="#2ECC71"/>
                <circle cx="60" cy="40" r="4" fill="#2ECC71"/>
                <circle cx="90" cy="10" r="4" fill="#2ECC71"/>
                <circle cx="120" cy="20" r="4" fill="#2ECC71"/>
            </g>
            
            <!-- Success Checkmark Circle -->
            <circle cx="250" cy="150" r="0" fill="#E8F0F6" opacity="0">
                <animate attributeName="r" from="0" to="70" dur="0.5s" begin="0.2s" fill="freeze"/>
                <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="0.2s" fill="freeze"/>
            </circle>
            
            <!-- Checkmark -->
            <path d="M220,150 L240,170 L280,130" stroke="#27AE60" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0">
                <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="0.7s" fill="freeze"/>
            </path>
            
            <!-- Success Text -->
            <text x="250" y="240" font-family="Arial" font-size="20" font-weight="bold" text-anchor="middle" fill="#27AE60" opacity="0">
                Analysis Complete!
                <animate attributeName="opacity" from="0" to="1" dur="0.5s" begin="1.2s" fill="freeze"/>
            </text>
            
            <!-- Confetti Animation -->
            <g opacity="0">
                <animate attributeName="opacity" from="0" to="1" dur="0.3s" begin="1.5s" fill="freeze"/>
                <!-- Confetti Pieces -->
                <circle cx="100" cy="100" r="5" fill="#3498DB">
                    <animate attributeName="cy" from="0" to="300" dur="2s" begin="1.5s" repeatCount="indefinite"/>
                </circle>
                <circle cx="150" cy="50" r="5" fill="#E74C3C">
                    <animate attributeName="cy" from="0" to="300" dur="2.2s" begin="1.6s" repeatCount="indefinite"/>
                </circle>
                <circle cx="200" cy="70" r="5" fill="#F1C40F">
                    <animate attributeName="cy" from="0" to="300" dur="1.8s" begin="1.7s" repeatCount="indefinite"/>
                </circle>
                <circle cx="300" cy="30" r="5" fill="#2ECC71">
                    <animate attributeName="cy" from="0" to="300" dur="2.1s" begin="1.8s" repeatCount="indefinite"/>
                </circle>
                <circle cx="350" cy="90" r="5" fill="#9B59B6">
                    <animate attributeName="cy" from="0" to="300" dur="1.9s" begin="1.9s" repeatCount="indefinite"/>
                </circle>
                <rect x="120" y="20" width="10" height="10" fill="#E67E22">
                    <animate attributeName="cy" from="0" to="300" dur="2s" begin="1.5s" repeatCount="indefinite"/>
                </rect>
                <rect x="280" y="40" width="10" height="10" fill="#16A085">
                    <animate attributeName="cy" from="0" to="300" dur="2.2s" begin="1.6s" repeatCount="indefinite"/>
                </rect>
            </g>
        </svg>
        """
    
    else:
        # Default SVG if no matching type is found
        svg_code = """
        <svg width="500" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect width="500" height="300" fill="#f8f9fa" rx="10" ry="10"/>
            <text x="250" y="150" font-family="Arial" font-size="20" text-anchor="middle" fill="#666">
                Financial ML Pipeline
            </text>
        </svg>
        """
    
    # Convert SVG to base64
    svg_bytes = svg_code.encode()
    encoded = base64.b64encode(svg_bytes).decode()
    
    return encoded