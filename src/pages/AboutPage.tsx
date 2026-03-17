const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 sm:py-16 px-4 sm:px-6">
      <div className="mb-8 sm:mb-10 mt-12 sm:mt-16 text-center">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-foreground mb-3 sm:mb-4">
          About MedMate
        </h1>
        <p className="text-sm sm:text-base md:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed space-y-4">
          <span className="block">
            MedMate ML is a user-friendly web application that helps predict health risks like dementia and diabetes. 
            It provides quick assessments and easy-to-understand results, making health monitoring accessible for everyone.
          </span>
          
          <span className="block mt-4">
            Designed for both individuals and healthcare professionals, MedMate leverages advanced machine learning 
            to deliver reliable predictions based on real clinical data. The platform offers a modern, intuitive interface, 
            ensuring a seamless experience on any device.
          </span>
          
          <span className="block mt-4">
            With MedMate, users can take proactive steps toward their well-being, gain valuable insights into their 
            health status, and make informed decisions—all with privacy and security in mind.
          </span>
        </p>
      </div>
    </div>
  );
};

export default AboutPage;