import logo from '../../senti-logo.png'
import './logo.css'

export default function Logo() {
    return (
        <div className="Logo">
            <img src={logo} className='Logo-Img'/>
            <p className='Logo-Text'>Svelti</p>
        </div>
    );
}