import ImageInput from './ImageInput';
import Button from "./Button";
import Predictions from './Predictions';

import { useContext, useState } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";


export default function Form(){
    let [image, setImage] = useState(null);

    let style;
    const designs = useContext(DesignsContext);
    const themes = useContext(ThemeContext);
    const { design, theme } = themes;
    
    // sometimes themes context will contain only the design 
    // and not the theme key so check if theme key is in themes
    if('theme' in themes){
        style = designs[design][theme];
    }else{
        style = designs[design];
    }

    // send a post request and retrieve the response and set 
    // the state of the following states for the alert component
    let [response, setResponse] = useState(null);
    let [msgStatus, setMsgStatus] = useState();
    let [errorType, setErrorType] = useState(null);
    const handleSubmit = async (event) => {
        try {
            event.preventDefault();
            const form_data = new FormData();
            form_data.append('image', image);

            // once data is validated submitted and then extracted
            // reset form components form element
            setImage(null);

            // send here the data from the contact component to 
            // the backend proxy server
            // // for development
            const url = 'http://127.0.0.1:5000/predict';
            // for production
            // const url = 'https://project-alexander.vercel.app/predict';

            const resp = await fetch(url, {
                'method': 'POST',
                'body': form_data,
            });

            const data = await resp.json();
            setResponse(data);

            // if response.status is 200 then that means contact information
            // has been successfully sent to the email.js api
            if(resp.status === 200){
                setMsgStatus("success");
                console.log(`message has been sent with code ${resp.status}`);

            }else{
                setMsgStatus("failure");
                console.log(`message submission unsucessful. Response status '${resp.status}' occured`);
            }

        }catch(error){
            setMsgStatus("denied");
            setErrorType(error);
            console.log(`Submission denied. Error '${error}' occured`);
        }
    };

    console.log(`response: ${response}`);
    console.log(`message status: ${msgStatus}`);
    console.log(`error type: ${errorType}`)

    return (
        <FormInputsContext.Provider value={{
            image, setImage,
            response,
            handleSubmit,
        }}>
            <div className="form-container">
                <form
                    className={`form ${design}`}
                    style={style}
                    method="POST"
                >
                    <ImageInput/>
                    <Button/>
                    <Predictions/>                   
                </form>
                <div className={`alert ${msgStatus !== undefined ? 'show' : ''}`} onClick={(event) => {
                    // remove class from alert container to hide it again
                    event.target.classList.remove('show');
                
                    // reset msg_status to undefined in case of another submission
                    setMsgStatus(undefined);
                }}>
                    <div className="alert-wrapper">
                        {msgStatus === "success" || msgStatus === "failed" ? 
                        <span className="alert-message">Message has been sent with code {response?.status}</span> : 
                        <span className="alert-message">Submission denied. Error {errorType?.message} occured</span>}
                    </div>
                </div>
            </div>
        </FormInputsContext.Provider>
    );
}