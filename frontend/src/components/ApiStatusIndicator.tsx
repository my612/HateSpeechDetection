
import React, { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';
import { checkApiHealth } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

const ApiStatusIndicator: React.FC = () => {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const { toast } = useToast();

  const checkConnection = async () => {
    if (isChecking) return;
    
    setIsChecking(true);
    try {
      const isHealthy = await checkApiHealth();
      setIsConnected(isHealthy);
      
      if (!isHealthy) {
        toast({
          title: "API Connection Error",
          description: "Cannot connect to the hate speech detection API. Please check if the API server is running.",
          variant: "destructive",
        });
      }
    } catch (error) {
      setIsConnected(false);
      toast({
        title: "API Connection Error",
        description: "Failed to connect to the API server. Please check your network connection.",
        variant: "destructive",
      });
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();
    
    // Periodically check connection
    const interval = setInterval(checkConnection, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);

  if (isConnected === null) {
    return (
      <div className="flex items-center text-gray-400 text-xs">
        <span className="animate-pulse">Checking API connection...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center text-xs gap-1" onClick={checkConnection}>
      {isConnected ? (
        <>
          <CheckCircle className="w-3 h-3 text-green-500" />
          <span className="text-green-600">API Connected</span>
        </>
      ) : (
        <>
          <AlertCircle className="w-3 h-3 text-red-500" />
          <span className="text-red-600">API Disconnected</span>
        </>
      )}
    </div>
  );
};

export default ApiStatusIndicator;
