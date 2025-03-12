// Navigation scroll behavior with improved smooth scrolling
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.navigation a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                // Get the current position of the element relative to the viewport
                const elementPosition = targetElement.getBoundingClientRect().top;
                
                // Add the current scroll position to get the position relative to the document
                const offsetPosition = elementPosition + window.pageYOffset - 60;
                
                // Smooth scroll to the element
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
                
                // Highlight the section briefly
                targetElement.classList.add('highlight');
                setTimeout(() => {
                    targetElement.classList.remove('highlight');
                }, 1500);
            }
        });
    });
    
    // Add active state to navigation links based on scroll position
    window.addEventListener('scroll', function() {
        const scrollPosition = window.scrollY;
        
        // Define sections to track
        const sections = ['step1', 'step4', 'step7', 'step9'];
        
        // Find which section is currently in view
        let currentSection = sections[0];
        
        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            if (section) {
                const sectionTop = section.getBoundingClientRect().top + window.pageYOffset - 100;
                if (scrollPosition >= sectionTop) {
                    currentSection = sectionId;
                }
            }
        });
        
        // Update active navigation link
        document.querySelectorAll('.navigation a').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    });
});