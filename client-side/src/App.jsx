import { ThemeContext } from './contexts/ThemeContext';
import { DesignsProvider } from './contexts/DesignsContext';

import Correspondence from './components/Correspondence';
import './App.css';

function App() {

  return (
    <DesignsProvider>
      <ThemeContext.Provider value={{design: "sharp-minimal", theme: 'dark'}}>
        <Correspondence/>
      </ThemeContext.Provider>  
    </DesignsProvider> 
  );
}

export default App
