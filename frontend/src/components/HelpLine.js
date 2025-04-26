import React from 'react';
import SidePanel from './SidePanel';
import './HelpLine.css';

const HelpLine = () => {
  return (
    <div className="helpline-container">
      {/* Use SidePanel */}
      <SidePanel username="User" />

      {/* Main Content */}
      <div className="main-content">
        <h1 className="helpline-heading">Help Line</h1>
        <hr />


        {/* Banner */}
        <div className="helpline-banner">
        <p>1926 - National Mental Health Helpline - National Institute of Mental Health, Sri Lanka.</p>
        </div>

        {/* Boxes with Helpline Details */}
        <div className="helpline-boxes">
          {/* Box 1 */}
          <div className="helpline-box">
            <h2>National Institute of Mental Health</h2>
            <p><b>Details: </b>Provide comprehensive and evidence based Mental Health Services appropriate to the Local Context through state of the art approaches to patient care, capacity building, advocacy, community engagement, multi-sector collaboration and research delivered by competent and reliable staff</p>
            <p><b>Email: </b> info@nimh.health.gov.lk</p>
            <p><b>Location: </b> National Institute of Mental Health, Mulleriyawa New Town</p>
            <ul>
              <li><b>National Mental Health Helpline:</b> 1926</li>
              <li><b>Day Treatment Centre:</b> +94112578556</li>
              <li><b>Dementia Hotline: </b>+94113140844</li>
            </ul>
            <a href="https://nimh.health.gov.lk/en/" target="_blank" rel="noopener noreferrer">
              Visit Website
            </a>
            {/* Map for Box 1 */}
            <div className="map-container">
              <iframe
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d7921.903741722606!2d79.94801327093464!3d6.910576196590596!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae24679b4d37615%3A0x9f7dc37e9b206e93!2sNational%20Institute%20of%20Mental%20Health!5e0!3m2!1sen!2slk!4v1631546438000!5m2!1sen!2slk"
                title="National Institute of Mental Health"
                allowFullScreen
                loading="lazy"
              ></iframe>
            </div>
          </div>

          {/* Box 2 */}
          <div className="helpline-box">
            <h2>National Council for Mental Health</h2>
            <p><b>Details: </b> The National Council for Mental Health - NCMH is a non profit non governmental organization, working towards the development of mental health and mental health care in Sri Lanka. Established in 1982, it was incorporated by an Act of Parliament in 1986 and is a registered charity with the Ministry of Finance. NCMH maintains the Borella Resource Centre and the Gorakana Residential Facility.</p>
            <ul>
              <li>
                <strong>Resource Centre:</strong>
                <br />
                No. 96/20, Kitulwatta Road, Colombo - 08. Sri Lanka
                <br />
                Hotline: +94 743 636 386, (+94) 112 685 960
              </li>
              <li>
                <strong>Residential Facility:</strong>
                <br />
                No. 115/2, Galkanuwa Rd., Gorakana, Panadura. Sri Lanka
                <br />
                Hotline: +94 768 262 682, +94 383 398 317
              </li>
            </ul>
            <a href="https://www.ncmh.lk/" target="_blank" rel="noopener noreferrer">
              Visit Website
            </a>
            {/* Map for Box 2 */}
            <div className="map-container">
              <iframe
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d31687.99173696736!2d79.8832992838153!3d6.91259019336106!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae25b98f65e2c1b%3A0xd45df109db7e4a7c!2sNational%20Council%20for%20Mental%20Health!5e0!3m2!1sen!2slk!4v1631546629000!5m2!1sen!2slk"
                title="National Council for Mental Health"
                allowFullScreen
                loading="lazy"
              ></iframe>
            </div>
          </div>

          {/* Box 3 */}
          <div className="helpline-box">
            <h2>Sri Lanka Sumithrayo</h2>
            <p><b>Details:</b> Sri Lanka Sumithrayo offers emotional support to those who are lonely, depressed, despairing, and in danger of taking their own lives. <br/>
            Providing Emotional support for people to develop coping skills and deal with everyday stress and strains of life
            </p>
            <p><b>Address: </b> 60/7, Horton Place, Colombo, Sri Lanka</p>
            <p><b>Email: </b> info.slsumithrayo@gmail.com</p>
            <p><b>Hotlines:</b> +94 707 308 308</p>
            <p><b>Whatsapp:</b> +94 767 520 620</p>            
            <a href="https://srilankasumithrayo.lk/" target="_blank" rel="noopener noreferrer">
              Visit Website
            </a>
            {/* Map for Box 3 */}
            <div className="map-container">
              <iframe
                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d31686.357214747917!2d79.85947678381544!3d6.914643593193073!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae25aaab2f5d79b%3A0x36739b246fc3de73!2sSri%20Lanka%20Sumithrayo!5e0!3m2!1sen!2slk!4v1631546817000!5m2!1sen!2slk"
                title="Sri Lanka Sumithrayo"
                allowFullScreen
                loading="lazy"
              ></iframe>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HelpLine;