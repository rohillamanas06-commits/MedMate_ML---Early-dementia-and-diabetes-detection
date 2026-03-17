import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

const AboutPage = () => {
  return (
    <div className="max-w-4xl mx-auto py-16 px-4">
      <div className="mb-10 mt-16 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-foreground mb-4">About MedMate</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          MedMate ML is a user-friendly web application that helps predict health risks like dementia and diabetes. It provides quick assessments and easy-to-understand results, making health monitoring accessible for everyone.<br /><br />
          Designed for both individuals and healthcare professionals, MedMate leverages advanced machine learning to deliver reliable predictions based on real clinical data. The platform offers a modern, intuitive interface, ensuring a seamless experience on any device.<br /><br />
          With MedMate, users can take proactive steps toward their well-being, gain valuable insights into their health status, and make informed decisions all with privacy and security in mind.
        </p>
      </div>
    </div>
  );
};

export default AboutPage;
